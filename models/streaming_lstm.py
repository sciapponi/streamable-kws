from pandas._config import using_nullable_dtypes
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

class Improved_Phi_LSTM_ATT_Streaming(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, n_fft=400, hop_length=160, matchbox={}, export_mode=False):
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

        # Core model components
        self.phi = MatchboxNetSkip(matchbox)
        # self.lstm = nn.LSTM(
        #     input_size=matchbox.get('base_filters', 32),
        #     hidden_size=hidden_dim,
        #     batch_first=True,
        #     bidirectional=False
        # )
        self.lstm = CustomLSTM(
            input_size=matchbox.get('base_filters', 32), # This is the feature dim after phi
            hidden_size=hidden_dim
        )
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.keyword_attention = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.hidden_dim = hidden_dim

    def forward(self, x, hidden_state=None, cell_state=None):
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

        # Process through model
        x = self.phi(x)
        x = x.permute(0, 2, 1).contiguous()  # (B, Time, Features)

        # Handle hidden and cell states for LSTM
        if hidden_state is None or cell_state is None:
            batch_size = x.size(0)
            hidden_state = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)
            cell_state = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)

        x, (new_hidden_state, new_cell_state) = self.lstm(x, (hidden_state, cell_state))
        x = self.projection(x)
        x, _ = self.keyword_attention(x)
        output = self.fc(x)

        return output, new_hidden_state, new_cell_state

    def export_onnx(self, save_path, input_shape=(1, 64, 50), simplify_model=True, use_dynamic_axes=True):
        """
        Optimized ONNX export method with optional simplification and dynamic axes for LSTM

        Args:
            save_path (str): Path to save the ONNX model
            input_shape (tuple): Shape of input spectrogram (batch_size, n_mels, time_steps)
            simplify_model (bool): Whether to simplify the exported model
            use_dynamic_axes (bool): Whether to use dynamic axes for variable input sizes
        """
        # Set to export mode and evaluation mode
        self.export_mode = True
        self.eval()

        # Store original device and move model to CPU for export
        original_device = next(self.parameters()).device
        cpu_model = self.to('cpu')

        # Validate input shape
        batch_size, n_mels, time_steps = input_shape
        if n_mels != self.n_mel_bins:
            raise ValueError(f"Expected {self.n_mel_bins} mel bins, got {n_mels}")

        # Prepare dummy inputs (on CPU) - LSTM needs both hidden and cell states
        spectrogram_input = torch.randn(batch_size, n_mels, time_steps)
        h0_input = torch.zeros(1, batch_size, self.hidden_dim)
        c0_input = torch.zeros(1, batch_size, self.hidden_dim)

        try:
            # Configure dynamic axes based on parameter
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
                print("Using dynamic axes for flexible input sizes")
            else:
                print("Using fixed input sizes (no dynamic axes)")

            # Initial export
            torch.onnx.export(
                cpu_model,                                          # model being run (on CPU)
                (spectrogram_input, h0_input, c0_input),           # model inputs (on CPU)
                save_path,                                          # where to save the model

                # Input and output names
                input_names=['spectrogram_input', 'hidden_state_input', 'cell_state_input'],
                output_names=['output_logits', 'hidden_state_output', 'cell_state_output'],

                # Dynamic axes (optional)
                dynamic_axes=dynamic_axes,

                # Optimization parameters
                opset_version=11,
                export_params=True,
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL,

                # Reduce verbosity
                verbose=False
            )

            # Verify the exported model
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            print(f"ONNX model exported successfully to {save_path}")

            # Optional model simplification
            if simplify_model:
                try:
                    # Simplify the model
                    simplified_model, check = simplify(onnx_model)

                    # Save simplified model
                    simplified_path = save_path.replace('.onnx', '_simplified.onnx')
                    onnx.save(simplified_model, simplified_path)

                    print(f"Model simplified. Simplified model saved to {simplified_path}")
                    print(f"Simplification check passed: {check}")

                    # Use simplified model for further checks
                    onnx_model = simplified_model
                    save_path = simplified_path
                except Exception as simplify_error:
                    print(f"Model simplification failed: {simplify_error}")

            # Print model sizes
            original_size = os.path.getsize(save_path) / 1024
            print(f"Model size: {original_size:.2f} KB")

            # Optional: Runtime verification
            ort_session = onnxruntime.InferenceSession(save_path)
            ort_inputs = {
                'spectrogram_input': spectrogram_input.numpy(),
                'hidden_state_input': h0_input.numpy(),
                'cell_state_input': c0_input.numpy()
            }
            ort_outs = ort_session.run(
                ['output_logits', 'hidden_state_output', 'cell_state_output'],
                ort_inputs
            )
            print("ONNX Runtime test passed.")
            print(f"Output shapes: {ort_outs[0].shape}, Hidden state: {ort_outs[1].shape}, Cell state: {ort_outs[2].shape}")

            # Move model back to original device
            self.to(original_device)

            return save_path

        except Exception as e:
            print(f"ONNX export failed: {e}")
            import traceback
            traceback.print_exc()

            # Move model back to original device
            self.to(original_device)

            return None

@hydra.main(version_base=None, config_path="../config", config_name="4_classes")
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)

    # Setup logging and configuration
    output_dir = HydraConfig.get().runtime.output_dir
    num_classes = 5
    hidden_dim = cfg.model.get("hidden_dim", 32)
    n_mel_bins = cfg.model.get("n_mel_bins", 40)
    input_time_steps = 18
    batch_size = 1

    # Model instantiation
    log.info("Instantiating model...")
    model = instantiate(cfg.model, num_classes=num_classes, export_mode=True)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters in model: {total_params}')

    model.eval()

    # Prepare dummy inputs - LSTM needs both hidden and cell states
    dummy_input = torch.randn(batch_size, n_mel_bins, input_time_steps)
    dummy_h0 = torch.zeros(1, batch_size, hidden_dim)
    dummy_c0 = torch.zeros(1, batch_size, hidden_dim)

    # Export paths
    export_path = f"{output_dir}/test_export.onnx"
    log.info(f"Exporting ONNX model to: {export_path}")

    # Export with optional dynamic axes (set to False to disable dynamic axes)
    # use_dynamic_axes = cfg.export.get("use_dynamic_axes", False)  # Default to True, can be overridden in config
    use_dynamic_axes = False
    exported_model_path = model.export_onnx(
        save_path=export_path,
        input_shape=(batch_size, n_mel_bins, input_time_steps),
        simplify_model=True,          # Enable simplification
        use_dynamic_axes=use_dynamic_axes  # Control dynamic axes usage
    )

    if exported_model_path:
        # Optional: Inspect model weights
        # onnx_model = onnx.load(exported_model_path)
        # print("\nModel Weight Datatypes:")
        # for tensor in onnx_model.graph.initializer:
        #     print(f"{tensor.name}: {tensor.data_type}")

        log.info("Running ONNX model with ONNX Runtime...")

        # ONNX Runtime inference - LSTM needs both states
        ort_session = onnxruntime.InferenceSession(exported_model_path)
        ort_inputs = {
            "spectrogram_input": dummy_input.numpy().astype(np.float32),
            "hidden_state_input": dummy_h0.numpy().astype(np.float32),
            "cell_state_input": dummy_c0.numpy().astype(np.float32)
        }

        ort_outputs = ort_session.run(["output_logits", "hidden_state_output", "cell_state_output"], ort_inputs)

        logits, new_h, new_c = ort_outputs
        log.info(f"ONNX output logits shape: {logits.shape}")
        log.info(f"ONNX output hidden state shape: {new_h.shape}")
        log.info(f"ONNX output cell state shape: {new_c.shape}")
        log.info(f"Top prediction class: {np.argmax(logits, axis=1)}")

if __name__ == "__main__":
    main()
