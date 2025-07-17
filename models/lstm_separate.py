import torch
import torch.nn as nn
import torchaudio
import onnx
import onnxruntime
from onnxsim import simplify
from modules import MatchboxNetSkip, AttentionLayer
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

class TwoStageModel:
    """Wrapper class to handle the two-stage inference"""
    def __init__(self, cnn_backbone, streaming_processor):
        self.cnn_backbone = cnn_backbone
        self.streaming_processor = streaming_processor

    def extract_features(self, spectrogram):
        """Extract CNN features from full spectrogram (18+ frames)"""
        with torch.no_grad():
            features = self.cnn_backbone(spectrogram)
        return features

    def process_streaming(self, feature_frame, h_prev, c_prev):
        """Process single feature frame through LSTM"""
        with torch.no_grad():
            output, h_new, c_new = self.streaming_processor(feature_frame, h_prev, c_prev)
        return output, h_new, c_new

def simplify_onnx_model(model_path, log):
    """Simplify ONNX model and save simplified version"""
    try:
        log.info(f"Simplifying {model_path}...")
        onnx_model = onnx.load(model_path)
        simplified_model, check = simplify(onnx_model)

        simplified_path = model_path.replace('.onnx', '_simplified.onnx')
        onnx.save(simplified_model, simplified_path)

        original_size = os.path.getsize(model_path) / 1024
        simplified_size = os.path.getsize(simplified_path) / 1024

        log.info(f"Model simplified: {original_size:.2f} KB -> {simplified_size:.2f} KB")
        log.info(f"Simplification check passed: {check}")
        log.info(f"Simplified model saved to {simplified_path}")

        return simplified_path
    except Exception as e:
        log.warning(f"Model simplification failed: {e}")
        return model_path

def export_separate_models(cfg: DictConfig):
    """Export CNN backbone and streaming processor separately"""
    log = logging.getLogger(__name__)
    output_dir = HydraConfig.get().runtime.output_dir

    num_classes = 5
    hidden_dim = cfg.model.get("hidden_dim", 32)
    n_mel_bins = cfg.model.get("n_mel_bins", 40)

    # CNN backbone parameters
    cnn_time_steps = 18  # Minimum frames needed for good CNN features
    cnn_batch_size = 1

    # Streaming processor parameters
    streaming_batch_size = 1
    cnn_feature_dim = cfg.model.matchbox.get('base_filters', 32)

    log.info("Creating CNN backbone...")
    cnn_backbone = CNNBackbone(
        n_mel_bins=n_mel_bins,
        matchbox=cfg.model.matchbox,
        export_mode=True
    )

    log.info("Creating streaming processor...")
    streaming_processor = StreamingProcessor(
        input_size=cnn_feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )

    # Load weights from original model if available
    if hasattr(cfg, 'checkpoint_path') and cfg.checkpoint_path:
        log.info(f"Loading weights from {cfg.checkpoint_path}")
        checkpoint = torch.load(cfg.checkpoint_path, map_location='cpu')

        # Extract CNN weights
        cnn_state_dict = {k.replace('phi.', ''): v for k, v in checkpoint.items() if k.startswith('phi.')}
        cnn_backbone.phi.load_state_dict(cnn_state_dict)

        # Extract LSTM + classifier weights
        streaming_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('lstm.'):
                new_key = k.replace('lstm.', 'lstm_cell.')
                streaming_state_dict[new_key] = v
            elif k.startswith(('projection.', 'keyword_attention.', 'fc.')):
                if k.startswith('keyword_attention.'):
                    new_key = k.replace('keyword_attention.', 'attention.')
                else:
                    new_key = k
                streaming_state_dict[new_key] = v

        streaming_processor.load_state_dict(streaming_state_dict, strict=False)

    # Set models to eval mode
    cnn_backbone.eval()
    streaming_processor.eval()

    # Export CNN backbone
    log.info("Exporting CNN backbone...")
    cnn_dummy_input = torch.randn(cnn_batch_size, n_mel_bins, cnn_time_steps)
    cnn_export_path = f"{output_dir}/cnn_backbone.onnx"

    torch.onnx.export(
        cnn_backbone,
        cnn_dummy_input,
        cnn_export_path,
        input_names=['spectrogram'],
        output_names=['features'],
        # dynamic_axes={
        #     'spectrogram': {0: 'batch_size', 2: 'time_steps'},
        #     'features': {0: 'batch_size', 1: 'time_steps'}
        # },
        opset_version=11,
        export_params=True,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        verbose=False
    )

    # Verify and simplify CNN export
    onnx_model = onnx.load(cnn_export_path)
    onnx.checker.check_model(onnx_model)
    log.info(f"CNN backbone exported to {cnn_export_path}")
    cnn_simplified_path = simplify_onnx_model(cnn_export_path, log)

    # Export streaming processor
    log.info("Exporting streaming processor...")
    streaming_dummy_input = torch.randn(streaming_batch_size, cnn_feature_dim)
    streaming_dummy_h = torch.zeros(streaming_batch_size, hidden_dim)
    streaming_dummy_c = torch.zeros(streaming_batch_size, hidden_dim)
    streaming_export_path = f"{output_dir}/streaming_processor.onnx"

    torch.onnx.export(
        streaming_processor,
        (streaming_dummy_input, streaming_dummy_h, streaming_dummy_c),
        streaming_export_path,
        input_names=['feature_frame', 'hidden_state', 'cell_state'],
        output_names=['output_logits', 'new_hidden_state', 'new_cell_state'],
        # dynamic_axes={
        #     'feature_frame': {0: 'batch_size'},
        #     'hidden_state': {0: 'batch_size'},
        #     'cell_state': {0: 'batch_size'},
        #     'output_logits': {0: 'batch_size'},
        #     'new_hidden_state': {0: 'batch_size'},
        #     'new_cell_state': {0: 'batch_size'}
        # },
        opset_version=11,
        export_params=True,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        verbose=False
    )

    # Verify and simplify streaming processor export
    onnx_model = onnx.load(streaming_export_path)
    onnx.checker.check_model(onnx_model)
    log.info(f"Streaming processor exported to {streaming_export_path}")
    streaming_simplified_path = simplify_onnx_model(streaming_export_path, log)

    # Test the separated models (use simplified versions)
    log.info("Testing separated models...")

    # Test CNN backbone
    ort_cnn = onnxruntime.InferenceSession(cnn_simplified_path)
    cnn_output = ort_cnn.run(['features'], {'spectrogram': cnn_dummy_input.numpy()})
    cnn_features = cnn_output[0]
    log.info(f"CNN features shape: {cnn_features.shape}")

    # Test streaming processor with first frame
    ort_streaming = onnxruntime.InferenceSession(streaming_simplified_path)
    first_frame = cnn_features[:, 0, :]  # Take first time step
    streaming_output = ort_streaming.run(
        ['output_logits', 'new_hidden_state', 'new_cell_state'],
        {
            'feature_frame': first_frame.astype(np.float32),
            'hidden_state': streaming_dummy_h.numpy().astype(np.float32),
            'cell_state': streaming_dummy_c.numpy().astype(np.float32)
        }
    )

    logits, new_h, new_c = streaming_output
    log.info(f"Streaming output shape: {logits.shape}")
    log.info(f"New hidden state shape: {new_h.shape}")
    log.info(f"New cell state shape: {new_c.shape}")

    # Print model sizes (both original and simplified)
    cnn_size = os.path.getsize(cnn_export_path) / 1024
    cnn_simplified_size = os.path.getsize(cnn_simplified_path) / 1024
    streaming_size = os.path.getsize(streaming_export_path) / 1024
    streaming_simplified_size = os.path.getsize(streaming_simplified_path) / 1024

    log.info(f"CNN backbone: {cnn_size:.2f} KB -> {cnn_simplified_size:.2f} KB (simplified)")
    log.info(f"Streaming processor: {streaming_size:.2f} KB -> {streaming_simplified_size:.2f} KB (simplified)")
    log.info(f"Total simplified size: {(cnn_simplified_size + streaming_simplified_size):.2f} KB")

    return cnn_simplified_path, streaming_simplified_path

@hydra.main(version_base=None, config_path="../config", config_name="4_classes")
def main(cfg: DictConfig):
    export_separate_models(cfg)

if __name__ == "__main__":
    main()
