import torch
import torch.nn as nn
# Import your custom modules directly or ensure they are discoverable
# If MatchboxNetSkip and AttentionLayer are in 'modules.py' in the same directory:
from modules import MatchboxNetSkip, AttentionLayer
import ai_edge_torch
import os

# Define your model class here, or import it if it's in a separate file accessible
# from this script's environment. For simplicity, I'm including it here.
# If your model is complex, you might put the class definition in a 'model_def.py'
# file and import it from both your main project and this export script.
class Improved_Phi_LSTM_ATT_Streaming(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, n_fft=400, hop_length=160, matchbox={}, export_mode=False):
        super().__init__()
        self.export_mode = export_mode
        self.n_mel_bins = n_mel_bins

        # In export_mode, these are not used, so no need for torchaudio to be installed in this env
        if not self.export_mode:
            # These will not be instantiated when export_mode is True
            # self.mel_spec = torchaudio.transforms.MelSpectrogram(...)
            # self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
            pass # Or raise an error if export_mode is accidentally False

        self.phi = MatchboxNetSkip(matchbox)
        self.lstm = nn.LSTM(
            input_size=matchbox.get('base_filters', 32),
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.keyword_attention = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden_state=None, cell_state=None):
        if not self.export_mode:
            # These are bypassed in export_mode
            # x = self.mel_spec(x)
            # x = self.amplitude_to_db(x)
            pass

        if x.dim() == 3:
            x = x.unsqueeze(1) # Add channel dimension if missing

        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        x = (x - mean) / std
        x = x.squeeze(1) # (B, Mels, Time)

        x = self.phi(x)
        x = x.permute(0, 2, 1).contiguous() # (B, Time, Features)

        # In export mode, hidden_state and cell_state MUST be provided as inputs.
        # The 'if None' logic is only for non-exported dynamic graphs.
        # For export, the dummy inputs take care of providing initial states.
        x, (new_hidden_state, new_cell_state) = self.lstm(x, (hidden_state, cell_state))
        x = self.projection(x)
        x, _ = self.keyword_attention(x)
        output = self.fc(x)

        return output, new_hidden_state, new_cell_state

def export_model_to_tflite(
    model_weights_path: str,
    output_tflite_path: str,
    num_classes: int = 10,
    n_mel_bins: int = 64,
    hidden_dim: int = 32,
    input_time_steps: int = 50, # Example input length for tracing
    batch_size: int = 1,
    matchbox_config: dict = None # Pass the matchbox config from Hydra
):
    print(f"Loading model from {model_weights_path}")
    # Instantiate the model in export mode
    if matchbox_config is None:
        matchbox_config = {} # Default empty if not provided

    model = Improved_Phi_LSTM_ATT_Streaming(
        num_classes=num_classes,
        n_mel_bins=n_mel_bins,
        hidden_dim=hidden_dim,
        matchbox=matchbox_config,
        export_mode=True # Crucial for bypassing torchaudio transforms
    )

    # Load trained weights
    model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
    model.eval() # Set to evaluation mode

    print("Preparing dummy inputs for export...")
    # Prepare dummy inputs (on CPU) - LSTM needs both hidden and cell states
    spectrogram_input = torch.randn(batch_size, n_mel_bins, input_time_steps)
    h0_input = torch.zeros(1, batch_size, hidden_dim)
    c0_input = torch.zeros(1, batch_size, hidden_dim)

    example_inputs = (spectrogram_input, h0_input, c0_input)

    print(f"Starting TFLite conversion to {output_tflite_path}...")
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_tflite_path)
        if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty string
            os.makedirs(output_dir)

        # Perform the conversion
        edge_model = ai_edge_torch.convert(model, example_inputs)
        edge_model.export(output_tflite_path) # Use the corrected export path

        print(f"TFLite model successfully exported to {output_tflite_path}")
        tflite_size = os.path.getsize(output_tflite_path) / 1024
        print(f"TFLite model size: {tflite_size:.2f} KB")

    except Exception as e:
        print(f"TFLite export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # --- Example Usage ---
    # In a real scenario, you'd replace these with actual paths and model parameters
    # from your trained Hydra experiment.
    # You might pass these as command-line arguments to this script.

    # Placeholder for where your trained model weights would be saved
    # For demonstration, let's create a dummy weight file:
    dummy_model = Improved_Phi_LSTM_ATT_Streaming(export_mode=False)
    # Save a dummy state_dict
    dummy_weights_path = "dummy_trained_model.pth"
    torch.save(dummy_model.state_dict(), dummy_weights_path)
    print(f"Created dummy weights at: {dummy_weights_path}")


    # Define where the TFLite model should be saved
    tflite_output_dir = "exported_tflite_models"
    os.makedirs(tflite_output_dir, exist_ok=True)
    tflite_output_file = os.path.join(tflite_output_dir, "improved_phi_lstm.tflite")

    # You'd load these from your Hydra config or arguments from the main training script
    # For example:
    # Assuming '4_classes.yaml' has these parameters:
    # model:
    #   num_classes: 4
    #   hidden_dim: 32
    #   n_mel_bins: 40
    #   matchbox:
    #     base_filters: 32
    #     num_blocks: 3
    #     ...
    export_model_to_tflite(
        model_weights_path=dummy_weights_path, # Replace with your actual trained model path
        output_tflite_path=tflite_output_file,
        num_classes=4, # From your config
        n_mel_bins=40, # From your config
        hidden_dim=32, # From your config
        input_time_steps=50, # The length of the spectrogram slice you want to trace
        batch_size=1, # Usually 1 for streaming inference on edge
        matchbox_config={'base_filters': 32, 'num_blocks': 3, 'input_channels': 40} # Match your Hydra config for matchbox
    )

    # Clean up dummy weight file
    os.remove(dummy_weights_path)
    print(f"Removed dummy weights at: {dummy_weights_path}")
