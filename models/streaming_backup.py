import torch
import torch.nn as nn
import torchaudio
import onnx
import onnxruntime
from modules import MatchboxNetSkip, AttentionLayer


class Improved_Phi_GRU_ATT_Streaming(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, n_fft=400, hop_length=160, matchbox={}, export_mode=False):
        super(Improved_Phi_GRU_ATT_Streaming, self).__init__()
        # Flag to determine if we're in export mode (ONNX-compatible) or training mode
        self.export_mode = export_mode
        # Keep the spectrogram computation for fast training
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mel_bins
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        # Core model architecture
        self.phi = MatchboxNetSkip(matchbox)

        # Using GRU instead of LSTM
        self.gru = nn.GRU(
            input_size=matchbox.get('base_filters', 32),
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )

        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.keyword_attention = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Track if we're in streaming mode (processing chunks)
        self.streaming = False
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden_state):
        """
        Forward pass that supports both training and streaming inference

        Args:
            x: Input spectrogram or audio waveform
            hidden_state: GRU hidden state, used in streaming mode

        Returns:
            output: Model output (classification logits)
            new_hidden_state: Updated hidden state for next chunk (in streaming mode)
        """
        # Different input handling based on mode
        if not self.export_mode:
            # Training mode: Compute spectrograms on the fly
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add channel dimension if needed
            # Compute mel spectrogram (this part won't be in ONNX)
            x = self.mel_spec(x)
            x = self.amplitude_to_db(x)

        # From here on, code is shared between training and export modes
        # In export mode, x is already a spectrogram from the dataset
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension if needed

        # Normalize input
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        x = (x - mean) / std
        x = x.squeeze(1)  # Remove channel dimension

        # Process through the backbone
        x = self.phi(x)
        x = x.permute(0, 2, 1).contiguous()

        # Handle the GRU with explicit hidden state
        if hidden_state is None:
            # Initialize hidden state if not provided
            batch_size = x.size(0)
            hidden_state = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)

        # Process through GRU with explicit hidden state tracking
        x, new_hidden_state = self.gru(x, hidden_state)

        x = self.projection(x)
        x, attention_weights = self.keyword_attention(x)
        output = self.fc(x)

        return output, new_hidden_state

    def export_onnx(self, save_path, input_shape=(1, 1, 64, 50)):
        """
        Export the model to ONNX format with support for streaming inference
        Args:
            save_path: Path to save the ONNX model
            input_shape: Shape of the input spectrogram (batch, channels, mels, time)
        """
        self.export_mode = True
        self.eval()

        # Example inputs for tracing
        spectrogram = torch.randn(input_shape)
        batch_size = input_shape[0]
        h0 = torch.zeros(1, batch_size, self.hidden_dim)

        # Create a temporary class for ONNX export with modified forward
        class ExportModel(torch.nn.Module):
            def __init__(self, original_model):
                super(ExportModel, self).__init__()
                self.model = original_model

            def forward(self, spectrogram, h0):
                # No default parameters - forces both inputs to be used
                return self.model(spectrogram, h0)

        export_model = ExportModel(self)

        # Export to ONNX with dynamic axes for variable sequence length
        torch.onnx.export(
            export_model,
            (spectrogram, h0),
            save_path,
            input_names=['spectrogram', 'h0'],
            output_names=['output', 'h_next'],
            # dynamic_axes={
            #     'spectrogram': {3: 'time_steps'},  # Variable time dimension
            #     'output': {0: 'batch_size'},
            #     'h_next': {1: 'batch_size'}
            # },
            verbose=True,
            opset_version=13
        )

        print(f"Model exported to {save_path}")
        # Verify the model
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model checked successfully!")

    def export_onnx(self, save_path, input_shape=(1, 64, 50)):
        """
        Export the model to ONNX format with support for streaming inference.

        Args:
            save_path: Path to save the ONNX model.
            input_shape: Shape of the input spectrogram (batch, mels, time_steps).
                         Note: Channel dim is added internally.
        """
        if self.export_mode is False:
            print("Warning: Model was not initialized in export_mode. Setting export_mode=True.")
            # Ideally, create a new instance for export or ensure relevant layers exist
            # For now, just set the flag. This might fail if mel_spec was needed but not created.
            self.export_mode = True
            # We might need to re-initialize parts of the model if export_mode affects structure.
            # Assuming __init__ logic handles this correctly based on the flag.

        self.eval() # Set model to evaluation mode

        # Example inputs for tracing matching the expected ONNX input shapes
        # Input 1: Spectrogram (batch, mels, time_steps)
        batch_size, n_mels, time_steps = input_shape
        if n_mels != self.n_mel_bins:
            raise ValueError(f"Input shape n_mels ({n_mels}) doesn't match model n_mel_bins ({self.n_mel_bins})")

        spectrogram_input = torch.randn(batch_size, n_mels, time_steps)

        # Input 2: Initial hidden state (num_layers * directions, batch, hidden_size)
        # For single-layer, non-bidirectional GRU, shape is (1, batch, hidden_dim)
        h0_input = torch.zeros(1, batch_size, self.hidden_dim)

        # Define input and output names explicitly
        input_names = ['spectrogram_input', 'hidden_state_input']
        output_names = ['output_logits', 'hidden_state_output']

        # Use the model directly if its forward pass matches the export signature
        # The current forward pass takes (x, hidden_state=None)
        # For export, we need a forward that *requires* both inputs.
        # Let's modify the main forward or use a wrapper. Using a wrapper is cleaner.

        class ExportWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                # Ensure the wrapped model is in export mode
                self.model.export_mode = True
                self.model.eval()

            def forward(self, spec_input, hidden_in):
                # This guarantees both inputs are traced in ONNX export
                out, h_next = self.model(spec_input, hidden_in)
                return out, h_next

        export_model = ExportWrapper(self)

        # Define dynamic axes for variable batch size and time steps
        dynamic_axes = {
            input_names[0]: {0: 'batch_size', 2: 'time_steps'}, # Spectrogram: Batch, Mels, Time
            input_names[1]: {1: 'batch_size'},                 # Hidden State In: 1, Batch, HiddenDim
            output_names[0]: {0: 'batch_size'},                # Output Logits: Batch, NumClasses
            output_names[1]: {1: 'batch_size'}                 # Hidden State Out: 1, Batch, HiddenDim
        }

        print(f"Attempting ONNX export to: {save_path}")
        print(f"Input shapes: {input_names[0]}: {list(spectrogram_input.shape)}, {input_names[1]}: {list(h0_input.shape)}")
        print(f"Output names: {output_names}")
        print(f"Dynamic axes: {dynamic_axes}")

        try:
            torch.onnx.export(
                export_model,
                (spectrogram_input, h0_input), # Tuple of inputs
                save_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=11, # Or try 12, 13
                verbose=True,
                export_params=True, # Ensure weights are saved
                #do_constant_folding=True, # Keep True unless it causes issues
            )
            print(f"ONNX export successful to {save_path}")

            # --- Verification Step ---
            print("Verifying exported ONNX model...")
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model check passed.")

            # Print actual input/output names from the loaded model
            actual_input_names = [inp.name for inp in onnx_model.graph.input]
            actual_output_names = [out.name for out in onnx_model.graph.output]
            print(f"Actual ONNX Input Names: {actual_input_names}")
            print(f"Actual ONNX Output Names: {actual_output_names}")

            # Compare with expected names
            if set(actual_input_names) != set(input_names):
                 print(f"WARNING: Exported input names {actual_input_names} do not match expected {input_names}")
            if set(actual_output_names) != set(output_names):
                 print(f"WARNING: Exported output names {actual_output_names} do not match expected {output_names}")

            # Optional: Verify with onnxruntime
            try:
                ort_session = onnxruntime.InferenceSession(save_path)
                ort_inputs = {
                    input_names[0]: spectrogram_input.numpy(),
                    input_names[1]: h0_input.numpy()
                }
                ort_outs = ort_session.run(output_names, ort_inputs)
                print("ONNX Runtime inference test successful.")
                print(f"Output shapes: Logits: {ort_outs[0].shape}, Hidden State: {ort_outs[1].shape}")
            except Exception as e:
                print(f"ONNX Runtime verification failed: {e}")

        except Exception as e:
            print(f"ONNX export failed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Optionally set the model back to training mode if needed
            # self.train()
            # self.export_mode = False # Reset flag if necessary
            pass
