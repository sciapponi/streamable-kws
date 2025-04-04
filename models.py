from modules import HighwayGRU, MatchboxNetSkip, AttentionLayer
import torch
import torch.nn as nn
import torchaudio

class Phi_HGRU(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, num_layers=1):
        super(Phi_HGRU, self).__init__()
        
        # Mel spectrogram transformation
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mel_bins
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # CNN Feature Extractor (MatchboxNet or equivalent small CNN)
        self.phi = MatchboxNetSkip(input_channels=n_mel_bins)  # Ensure it returns (batch, num_filters, seq_len)

        # GRU instead of SRNN
        self.gru = HighwayGRU(input_size=64, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention = AttentionLayer(hidden_dim)
        # Fully connected layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

        # Initialize GRU weights
        self._init_weights()

    def _init_weights(self):
        """Custom weight initialization for stability."""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:  # Apply Xavier Uniform to weights
                if len(param.shape) == 2:  # Ensure it's a weight matrix (2D)
                    nn.init.xavier_uniform_(param)
                else:
                    print(f"Skipping xavier_uniform for {name} due to non-2D shape {param.shape}")
            elif 'bias' in name:  # Apply zero initialization to biases
                nn.init.zeros_(param)

    def forward(self, x):
        # Preprocessing
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)
        
        x = self.mel_spec(x)
        x = self.amplitude_to_db(x)

        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, mel_bins, time)
        
        # Normalize
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        x = (x - mean) / std

        # CNN Feature extraction
        x = x.squeeze(1)  # (batch, mel_bins, time)
        x = self.phi(x)   # (batch, num_filters, seq_len)

        # Reshape for GRU (batch, seq_len, features)
        x = x.permute(0, 2, 1).contiguous()  

        # GRU forward pass
        x, _ = self.gru(x)  # Output shape: (batch, seq_len, hidden_dim)
        x, _ = self.attention(x)  # Apply attention to GRU output
        # Take last time step's output (for classification)
        # x = x[:, -1, :]  # (batch, hidden_dim)

        # Fully connected layers
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, num_classes)

        return x  

class Phi_GRU(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, num_layers=1):
        super(Phi_GRU, self).__init__()
        
        # Mel spectrogram transformation
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mel_bins
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # CNN Feature Extractor (MatchboxNet or equivalent small CNN)
        self.phi = MatchboxNetSkip(input_channels=n_mel_bins)  # Ensure it returns (batch, num_filters, seq_len)

        # GRU instead of SRNN
        self.gru = nn.GRU(input_size=64, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention = AttentionLayer(hidden_dim)
        # Fully connected layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

        # Initialize GRU weights
        self._init_weights()

    def _init_weights(self):
        """Custom weight initialization for stability."""
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # Preprocessing
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)
        
        x = self.mel_spec(x)
        x = self.amplitude_to_db(x)

        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, mel_bins, time)
        
        # Normalize
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        x = (x - mean) / std

        # CNN Feature extraction
        x = x.squeeze(1)  # (batch, mel_bins, time)
        x = self.phi(x)   # (batch, num_filters, seq_len)

        # Reshape for GRU (batch, seq_len, features)
        x = x.permute(0, 2, 1).contiguous()  

        # GRU forward pass
        x, _ = self.gru(x)  # Output shape: (batch, seq_len, hidden_dim)
        x, _ = self.attention(x)  # Apply attention to GRU output
        # Take last time step's output (for classification)
        # x = x[:, -1, :]  # (batch, hidden_dim)

        # Fully connected layers
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, num_classes)

        return x