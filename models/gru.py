from torch import nn
import torchaudio
from modules import MatchboxNetSkip, AttentionLayer


class Phi_GRU_ATT(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, n_fft=400, hop_length=160, matchbox={}, export_mode=False):
        super(Phi_GRU_ATT, self).__init__()

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
        self.gru = nn.GRU(
            input_size=matchbox.get('base_filters', 32),
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.keyword_attention = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Different input handling based on mode
        if not self.export_mode:
            # Training mode: Compute spectrograms on the fly (fast training)
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

        # Process through the rest of the model
        x = self.phi(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.gru(x)
        x = self.projection(x)
        x, attention_weights = self.keyword_attention(x)
        x = self.fc(x)

        return x
