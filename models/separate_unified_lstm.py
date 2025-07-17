import torch
import torch.nn as nn
import torchaudio
import onnx
import onnxruntime
from onnxsim import simplify
from modules import MatchboxNetSkip, AttentionLayer, CustomLSTM
import numpy as np
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
import logging
import os


class CNNBackbone(nn.Module):
    """CNN backbone that processes full spectrograms and outputs frame-by-frame features"""
    def __init__(self, n_mel_bins=64, n_fft=400, hop_length=160, matchbox={}, export_mode=False):
        super().__init__()
        self.export_mode = export_mode
        self.n_mel_bins = n_mel_bins

        # Conditional mel spectrogram transforms only for non-export mode
        if not self.export_mode:
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mel_bins
            )
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # CNN backbone
        self.phi = MatchboxNetSkip(matchbox)

    def forward(self, x):
        # Mel spectrogram processing only in non-export mode
        if not self.export_mode:
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (B, 1, T)
            x = self.mel_spec(x)
            x = self.amplitude_to_db(x)

            # Normalize input
            if x.dim() == 3:
                x = x.unsqueeze(1)  # Add channel dimension if missing
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True) + 1e-5
            x = (x - mean) / std
            x = x.squeeze(1)  # (B, Mels, Time)

        # Process through CNN
        x = self.phi(x)  # Output: (B, Features, Time)
        x = x.permute(0, 2, 1).contiguous()  # (B, Time, Features)

        return x

class LSTMCell(nn.Module):
    """Single LSTM cell for frame-by-frame processing"""
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Gates
        self.input_gate = nn.Linear(input_size, hidden_size, bias=bias)
        self.forget_gate = nn.Linear(input_size, hidden_size, bias=bias)
        self.cell_gate = nn.Linear(input_size, hidden_size, bias=bias)
        self.output_gate = nn.Linear(input_size, hidden_size, bias=bias)

        # Hidden state connections
        self.hidden_input_gate = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_forget_gate = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_cell_gate = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_output_gate = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x_t, h_prev, c_prev):
        """
        Process single timestep
        Args:
            x_t: Input at time t (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
            c_prev: Previous cell state (batch_size, hidden_size)
        Returns:
            h_t: New hidden state (batch_size, hidden_size)
            c_t: New cell state (batch_size, hidden_size)
        """
        i_t = self.sigmoid(self.input_gate(x_t) + self.hidden_input_gate(h_prev))
        f_t = self.sigmoid(self.forget_gate(x_t) + self.hidden_forget_gate(h_prev))
        g_t = self.tanh(self.cell_gate(x_t) + self.hidden_cell_gate(h_prev))
        o_t = self.sigmoid(self.output_gate(x_t) + self.hidden_output_gate(h_prev))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t

class StreamingProcessor(nn.Module):
    """LSTM + Attention + Classification head for single frame processing"""
    def __init__(self, input_size, hidden_dim=32, num_classes=10):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Single LSTM cell
        self.lstm_cell = LSTMCell(input_size, hidden_dim)

        # Projection and attention (simplified for single frame)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_t, h_prev, c_prev):
        """
        Process single frame
        Args:
            x_t: Single frame features (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
            c_prev: Previous cell state (batch_size, hidden_size)
        Returns:
            output: Classification logits (batch_size, num_classes)
            h_new: New hidden state (batch_size, hidden_size)
            c_new: New cell state (batch_size, hidden_size)
        """
        # LSTM cell forward
        h_new, c_new = self.lstm_cell(x_t, h_prev, c_prev)

        # Project and apply attention (treat single frame as sequence of length 1)
        projected = self.projection(h_new)
        projected = projected.unsqueeze(1)  # Add sequence dimension
        attended, _ = self.attention(projected)
        attended = attended.squeeze(1)  # Remove sequence dimension

        # Classification
        output = self.fc(attended)

        return output, h_new, c_new


class UnifiedModel(nn.Module):
    """
    Unified model that can be used for both training and two-stage export.
    During training, it processes full sequences efficiently.
    During export, components can be separated for streaming inference.
    """
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, n_fft=400,
                 hop_length=160, matchbox={}, export_mode=False, training_mode=True):
        super().__init__()
        self.export_mode = export_mode
        self.training_mode = training_mode
        self.n_mel_bins = n_mel_bins
        self.hidden_dim = hidden_dim
        self.feature_dim = matchbox.get('base_filters', 32)

        # CNN Backbone
        self.cnn_backbone = CNNBackbone(
            n_mel_bins=n_mel_bins,
            n_fft=n_fft,
            hop_length=hop_length,
            matchbox=matchbox,
            export_mode=export_mode
        )

        # For training: use efficient LSTM implementation
        if training_mode:
            self.lstm = CustomLSTM(
                input_size=self.feature_dim,
                hidden_size=hidden_dim
            )
        else:
            # For streaming: use frame-by-frame LSTM cell
            self.lstm_cell = LSTMCell(self.feature_dim, hidden_dim)

        # Shared components for both modes
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.keyword_attention = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Streaming processor (for export)
        self.streaming_processor = StreamingProcessor(
            input_size=self.feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

    def forward(self, x, hidden_state=None, cell_state=None):
        """
        Forward pass that works for both training and inference
        """
        # Extract CNN features
        cnn_features = self.cnn_backbone(x)  # (B, Time, Features)

        if self.training_mode:
            # Training mode: process full sequences efficiently
            return self._forward_training(cnn_features, hidden_state, cell_state)
        else:
            # Streaming mode: process frame by frame
            return self._forward_streaming(cnn_features, hidden_state, cell_state)

    def _forward_training(self, cnn_features, hidden_state=None, cell_state=None):
        """Efficient training forward pass"""
        # Handle hidden and cell states for LSTM
        if hidden_state is None or cell_state is None:
            batch_size = cnn_features.size(0)
            hidden_state = torch.zeros(1, batch_size, self.hidden_dim, device=cnn_features.device)
            cell_state = torch.zeros(1, batch_size, self.hidden_dim, device=cnn_features.device)

        # Process through LSTM
        lstm_out, (new_hidden_state, new_cell_state) = self.lstm(cnn_features, (hidden_state, cell_state))

        # Apply projection and attention
        projected = self.projection(lstm_out)
        attended, _ = self.keyword_attention(projected)

        # Classification
        output = self.fc(attended)

        return output, new_hidden_state, new_cell_state

    def _forward_streaming(self, cnn_features, hidden_state=None, cell_state=None):
        """Frame-by-frame streaming forward pass"""
        batch_size, seq_len, feature_dim = cnn_features.shape

        if hidden_state is None or cell_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_dim, device=cnn_features.device)
            cell_state = torch.zeros(batch_size, self.hidden_dim, device=cnn_features.device)

        outputs = []
        h_t, c_t = hidden_state.squeeze(0) if hidden_state.dim() == 3 else hidden_state, \
                   cell_state.squeeze(0) if cell_state.dim() == 3 else cell_state

        # Process each frame
        for t in range(seq_len):
            frame_features = cnn_features[:, t, :]  # (B, Features)
            output_t, h_t, c_t = self.streaming_processor(frame_features, h_t, c_t)
            outputs.append(output_t.unsqueeze(1))

        # Concatenate outputs
        output = torch.cat(outputs, dim=1)  # (B, Time, Classes)

        # Return with proper dimensions for compatibility
        final_hidden = h_t.unsqueeze(0) if h_t.dim() == 2 else h_t
        final_cell = c_t.unsqueeze(0) if c_t.dim() == 2 else c_t

        return output, final_hidden, final_cell

    def sync_streaming_weights(self):
        """
        Synchronize weights between training LSTM and streaming LSTM cell
        Call this after training to prepare for streaming export
        """
        if hasattr(self, 'lstm') and hasattr(self, 'streaming_processor'):
            # Copy LSTM weights to streaming processor LSTM cell
            lstm_state_dict = self.lstm.state_dict()
            streaming_state_dict = {}

            # Map CustomLSTM weights to LSTMCell weights
            for key, value in lstm_state_dict.items():
                if 'weight_ih' in key or 'bias_ih' in key:
                    # Input-to-hidden weights
                    gate_name = key.split('_')[2] if len(key.split('_')) > 2 else key.split('_')[1]
                    if 'weight' in key:
                        new_key = f'lstm_cell.{gate_name}_gate.weight'
                    else:
                        new_key = f'lstm_cell.{gate_name}_gate.bias'
                elif 'weight_hh' in key or 'bias_hh' in key:
                    # Hidden-to-hidden weights
                    gate_name = key.split('_')[2] if len(key.split('_')) > 2 else key.split('_')[1]
                    if 'weight' in key:
                        new_key = f'lstm_cell.hidden_{gate_name}_gate.weight'
                    else:
                        new_key = f'lstm_cell.hidden_{gate_name}_gate.bias'
                else:
                    continue

                streaming_state_dict[new_key] = value

            # Copy projection, attention, and classifier weights
            for name, param in self.named_parameters():
                if name.startswith(('projection.', 'keyword_attention.', 'fc.')):
                    if name.startswith('keyword_attention.'):
                        new_name = name.replace('keyword_attention.', 'attention.')
                    else:
                        new_name = name
                    streaming_state_dict[new_name] = param.data

            # Load weights into streaming processor
            self.streaming_processor.load_state_dict(streaming_state_dict, strict=False)

    def switch_to_streaming_mode(self):
        """Switch model to streaming mode for export"""
        self.training_mode = False
        self.sync_streaming_weights()

    def switch_to_training_mode(self):
        """Switch model back to training mode"""
        self.training_mode = True

    def export_separate_models(self, output_dir, simplify_models=True):
        """
        Export CNN backbone and streaming processor as separate ONNX models
        """

        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        self.switch_to_streaming_mode()
        self.eval()

        # Export parameters
        cnn_time_steps = 18
        batch_size = 1

        # Get the device the model is on
        model_device = next(self.parameters()).device
        print(f"Model is on device: {model_device}")

        # Move model to CPU for export
        self.cpu()

        # Export CNN backbone
        print("Exporting CNN backbone...")
        cnn_dummy_input = torch.randn(batch_size, self.n_mel_bins, cnn_time_steps)
        cnn_export_path = f"{output_dir}/cnn_backbone.onnx"

        # Set CNN to export mode
        self.cnn_backbone.export_mode = True

        try:
            torch.onnx.export(
                self.cnn_backbone,
                cnn_dummy_input,
                cnn_export_path,
                input_names=['spectrogram'],
                output_names=['features'],
                opset_version=11,
                export_params=True,
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL,
                verbose=False
            )

            # Verify CNN export
            onnx_model = onnx.load(cnn_export_path)
            onnx.checker.check_model(onnx_model)
            print(f"CNN backbone exported to {cnn_export_path}")

            # Simplify CNN model if requested
            if simplify_models:
                cnn_export_path = self._simplify_onnx_model(cnn_export_path)

        except Exception as e:
            print(f"CNN export failed: {e}")
            # Move model back to original device before returning
            self.to(model_device)
            raise e

        # Export streaming processor
        print("Exporting streaming processor...")
        streaming_dummy_input = torch.randn(batch_size, self.feature_dim)
        streaming_dummy_h = torch.zeros(batch_size, self.hidden_dim)
        streaming_dummy_c = torch.zeros(batch_size, self.hidden_dim)
        streaming_export_path = f"{output_dir}/streaming_processor.onnx"

        try:
            torch.onnx.export(
                self.streaming_processor,
                (streaming_dummy_input, streaming_dummy_h, streaming_dummy_c),
                streaming_export_path,
                input_names=['feature_frame', 'hidden_state', 'cell_state'],
                output_names=['output_logits', 'new_hidden_state', 'new_cell_state'],
                opset_version=11,
                export_params=True,
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL,
                verbose=False
            )

            # Verify streaming processor export
            onnx_model = onnx.load(streaming_export_path)
            onnx.checker.check_model(onnx_model)
            print(f"Streaming processor exported to {streaming_export_path}")

            # Simplify streaming model if requested
            if simplify_models:
                streaming_export_path = self._simplify_onnx_model(streaming_export_path)

            # Test the exported models
            self._test_exported_models(cnn_export_path, streaming_export_path,
                                     cnn_dummy_input, streaming_dummy_h, streaming_dummy_c)

        except Exception as e:
            print(f"Streaming processor export failed: {e}")
            # Move model back to original device before returning
            self.to(model_device)
            raise e

        finally:
            # Always move model back to original device
            self.to(model_device)

        return cnn_export_path, streaming_export_path

    def export_unified_model(self, save_path, input_shape=(1, 64, 50), simplify_model=True, use_dynamic_axes=False):
        """
        Export as unified ONNX model (compatible with your second approach)
        """
        # Switch to streaming mode for export compatibility
        self.switch_to_streaming_mode()
        self.export_mode = True
        self.eval()

        # Get the device the model is on and move to CPU for export
        original_device = next(self.parameters()).device
        print(f"Moving model from {original_device} to CPU for export")
        cpu_model = self.to('cpu')

        batch_size, n_mels, time_steps = input_shape
        if n_mels != self.n_mel_bins:
            raise ValueError(f"Expected {self.n_mel_bins} mel bins, got {n_mels}")

        # Prepare dummy inputs (these will be on CPU by default)
        spectrogram_input = torch.randn(batch_size, n_mels, time_steps)
        h0_input = torch.zeros(1, batch_size, self.hidden_dim)
        c0_input = torch.zeros(1, batch_size, self.hidden_dim)

        try:
            # Configure dynamic axes
            dynamic_axes = None
            if use_dynamic_axes:
                dynamic_axes = {
                    'spectrogram_input': {0: 'batch_size', 2: 'time_steps'},
                    'hidden_state_input': {1: 'batch_size'},
                    'cell_state_input': {1: 'batch_size'},
                    'output_logits': {0: 'batch_size'},
                    'hidden_state_output': {1: 'batch_size'},
                    'cell_state_output': {1: 'batch_size'}
                }

            # Export unified model
            torch.onnx.export(
                cpu_model,
                (spectrogram_input, h0_input, c0_input),
                save_path,
                input_names=['spectrogram_input', 'hidden_state_input', 'cell_state_input'],
                output_names=['output_logits', 'hidden_state_output', 'cell_state_output'],
                dynamic_axes=dynamic_axes,
                opset_version=11,
                export_params=True,
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL,
                verbose=False
            )

            # Verify the exported model
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            print(f"Unified ONNX model exported successfully to {save_path}")

            # Simplify if requested
            if simplify_model:
                save_path = self._simplify_onnx_model(save_path)

            # Test the model
            self._test_unified_model(save_path, spectrogram_input, h0_input, c0_input)

            # Move back to original device
            self.to(original_device)
            return save_path

        except Exception as e:
            print(f"ONNX export failed: {e}")
            import traceback
            traceback.print_exc()
            self.to(original_device)
            return None

    def _simplify_onnx_model(self, model_path):
        """Simplify ONNX model and save simplified version"""
        try:
            print(f"Simplifying {model_path}...")
            onnx_model = onnx.load(model_path)
            simplified_model, check = simplify(onnx_model)

            # Create new path in a separate "_simplified" directory
            dir_path, file_name = os.path.split(model_path)
            base_name, _ = os.path.splitext(file_name)

            simplified_dir = dir_path + "_simplified"

            # ✅ Ensure the directory exists
            os.makedirs(simplified_dir, exist_ok=True)

            simplified_path = os.path.join(simplified_dir, base_name + "_simplified.onnx")

            # Save the simplified model
            onnx.save(simplified_model, simplified_path)

            original_size = os.path.getsize(model_path) / 1024
            simplified_size = os.path.getsize(simplified_path) / 1024

            print(f"Model simplified: {original_size:.2f} KB -> {simplified_size:.2f} KB")
            print(f"Simplification check passed: {check}")
            return simplified_path

        except Exception as e:
            print(f"Model simplification failed: {e}")
            return model_path


    def _test_exported_models(self, cnn_path, streaming_path, cnn_input, h_state, c_state):
        """Test the separated models"""
        print("Testing separated models...")

        # Test CNN backbone
        ort_cnn = onnxruntime.InferenceSession(cnn_path)
        cnn_output = ort_cnn.run(['features'], {'spectrogram': cnn_input.numpy()})
        cnn_features = cnn_output[0]
        print(f"CNN features shape: {cnn_features.shape}")

        # Test streaming processor with first frame
        ort_streaming = onnxruntime.InferenceSession(streaming_path)
        first_frame = cnn_features[:, 0, :]
        streaming_output = ort_streaming.run(
            ['output_logits', 'new_hidden_state', 'new_cell_state'],
            {
                'feature_frame': first_frame.astype(np.float32),
                'hidden_state': np.atleast_2d(h_state.numpy().astype(np.float32)),
                'cell_state': np.atleast_2d(c_state.numpy().astype(np.float32))
            }
        )

        logits, new_h, new_c = streaming_output
        print(f"Streaming output shape: {logits.shape}")
        print("Separated models test passed!")

    def _test_unified_model(self, model_path, spec_input, h_input, c_input):
        """Test the unified model"""
        print("Testing unified model...")

        ort_session = onnxruntime.InferenceSession(model_path)
        ort_inputs = {
            'spectrogram_input': spec_input.numpy(),
            'hidden_state_input': h_input.numpy(),
            'cell_state_input': c_input.numpy()
        }
        ort_outs = ort_session.run(
            ['output_logits', 'hidden_state_output', 'cell_state_output'],
            ort_inputs
        )
        print(f"Output shapes: {ort_outs[0].shape}, Hidden: {ort_outs[1].shape}, Cell: {ort_outs[2].shape}")
        print("Unified model test passed!")


class Phi_LSTM_ATT(UnifiedModel):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, n_fft=400,
                 hop_length=160, matchbox={}, export_mode=False):
        super().__init__(
            num_classes=num_classes,
            n_mel_bins=n_mel_bins,
            hidden_dim=hidden_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            matchbox=matchbox,
            export_mode=export_mode,
            training_mode=True  # Always start in training mode
        )

    # def forward(self, x, hidden_state=None, cell_state=None):
    #     return self(x, hidden_state, cell_state)

    # def export_unified_model(self, save_path, input_shape=(1, 64, 50), simplify_model=True, use_dynamic_axes=True):
    #     return self.core.export_unified_model(save_path, input_shape, simplify_model, use_dynamic_axes)

    # def switch_to_streaming_mode(self):
    #     self.core.switch_to_streaming_mode()

    def export_onnx(self, save_path, input_shape, simplify_models=True):
        self.export_separate_models(save_path, simplify_models)

# Training-compatible wrapper that behaves like your original model
class Improved_Phi_LSTM_ATT_Streaming(UnifiedModel):
    """
    Wrapper class for backward compatibility with your training code
    """
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, n_fft=400,
                 hop_length=160, matchbox={}, export_mode=False):
        super().__init__(
            num_classes=num_classes,
            n_mel_bins=n_mel_bins,
            hidden_dim=hidden_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            matchbox=matchbox,
            export_mode=export_mode,
            training_mode=True  # Always start in training mode
        )

    def export_onnx(self, save_path, input_shape=(1, 64, 50), simplify_model=True, use_dynamic_axes=True):
        """Backward compatibility method"""
        return self.export_unified_model(save_path, input_shape, simplify_model, use_dynamic_axes)

# Example usage functions
@hydra.main(version_base=None, config_path="../config", config_name="4_classes")
def main_separate_export(cfg: DictConfig):
    """Export as separate models (CNN + Streaming)"""
    log = logging.getLogger(__name__)
    output_dir = HydraConfig.get().runtime.output_dir

    # Model parameters
    num_classes = 5
    hidden_dim = cfg.model.get("hidden_dim", 32)
    n_mel_bins = cfg.model.get("n_mel_bins", 40)

    # Create model
    model = UnifiedModel(
        num_classes=num_classes,
        n_mel_bins=n_mel_bins,
        hidden_dim=hidden_dim,
        matchbox=cfg.model.matchbox,
        training_mode=True  # Start in training mode
    )

    # Load trained weights if available
    if hasattr(cfg, 'checkpoint_path') and cfg.checkpoint_path:
        log.info(f"Loading weights from {cfg.checkpoint_path}")
        checkpoint = torch.load(cfg.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

    # Export as separate models
    cnn_path, streaming_path = model.export_separate_models(output_dir, simplify_models=True)
    log.info(f"Models exported: {cnn_path}, {streaming_path}")

@hydra.main(version_base=None, config_path="../config", config_name="4_classes")
def main_unified_export(cfg: DictConfig):
    """Export as unified model (backward compatibility)"""
    log = logging.getLogger(__name__)
    output_dir = HydraConfig.get().runtime.output_dir

    # Model parameters
    num_classes = 5
    hidden_dim = cfg.model.get("hidden_dim", 32)
    n_mel_bins = cfg.model.get("n_mel_bins", 40)

    # Create model using compatibility wrapper
    model = Improved_Phi_LSTM_ATT_Streaming(
        num_classes=num_classes,
        n_mel_bins=n_mel_bins,
        hidden_dim=hidden_dim,
        matchbox=cfg.model.matchbox,
        export_mode=True
    )

    # Export unified model
    export_path = f"{output_dir}/unified_model.onnx"
    exported_path = model.export_onnx(
        save_path=export_path,
        input_shape=(1, n_mel_bins, 18),
        simplify_model=True,
        use_dynamic_axes=False
    )

    if exported_path:
        log.info(f"Unified model exported to {exported_path}")

if __name__ == "__main__":
    # Choose which export method to use
    mode = "separate"
    if mode == "separate":
        main_separate_export()
    else:
        main_unified_export()
