from modules import HighwayGRU, MatchboxNetSkip, AttentionLayer, StatefulRNNLayer, FocusedAttention, StatefulGRU, LightConsonantEnhancer
import torch
import torch.nn as nn
import torchaudio

class Improved_Phi_FC_Hybrid(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32):
        super(Improved_Phi_FC_Hybrid, self).__init__()
        # Keep your original mel spectrogram components
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mel_bins
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        # The CNN backbone
        self.phi = MatchboxNetSkip(input_channels=n_mel_bins)  # Expects n_mel_bins channels
        
        # First use a single GRU layer (more parameter efficient than multiple)
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False 
        )
        
        self.projection = nn.Linear(hidden_dim, hidden_dim)

        
        # Small self-attention to focus on keywords
        self.keyword_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,  # Single head for efficiency
            batch_first=True
        )
        
        # Add consonant enhancer (minimal parameters)
        self.consonant_enhancer = LightConsonantEnhancer(hidden_dim)
        # Keep your original attention
        self.attention = FocusedAttention(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
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
        # Remove the dimension we added
        x = x.squeeze(1)  # [batch, n_mels, time]
        # Now x should be correctly shaped for phi: [batch, n_mels, time]
        x = self.phi(x)  # Output shape: [batch, 64, time]
        x = x.permute(0, 2, 1).contiguous()  # [batch, time, 64]
        
        # GRU layer for sequential processing
        x, _ = self.gru(x)  # [batch, time, hidden_dim*2] (bidirectional)
        x = self.projection(x)  # [batch, time, hidden_dim]
        
        # Small attention layer focused on keywords
        residual = x
        attn_output, _ = self.keyword_attention(
            query=x,
            key=x,
            value=x
        )
        x = residual + attn_output  # Residual connection
        
        # Enhance consonant features
        x = self.consonant_enhancer(x)
        x, _ = self.attention(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
class Improved_Phi_FC_Attention(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, num_heads=2):
        super(Improved_Phi_FC_Attention, self).__init__()
        # Keep your original mel spectrogram components
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mel_bins
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        # The CNN backbone
        self.phi = MatchboxNetSkip(input_channels=n_mel_bins)  # Expects n_mel_bins channels
        
        # Replace RNN with lightweight Multi-head Attention
        self.input_projection = nn.Linear(64, hidden_dim)
        
        self.hidden_dim = hidden_dim
        # Reduced number of heads and smaller feed-forward dim
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Simplified feed-forward network with smaller expansion factor
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),  # Reduced from 4x to 2x expansion
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Single combined normalization layer
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Add consonant enhancer (minimal parameters)
        self.consonant_enhancer = LightConsonantEnhancer(hidden_dim)
        # Keep your original attention
        self.attention = FocusedAttention(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
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
        # Remove the dimension we added
        x = x.squeeze(1)  # [batch, n_mels, time]
        # Now x should be correctly shaped for phi: [batch, n_mels, time]
        x = self.phi(x)  # Output shape: [batch, 64, time]
        x = x.permute(0, 2, 1).contiguous()  # [batch, time, 64]
        
        # Project to hidden dimension
        x = self.input_projection(x)  # [batch, time, hidden_dim]
        
        # Add positional information using relative position
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        positions = positions.unsqueeze(-1).float() / seq_len  # Simple normalized position [0-1]
        x = torch.cat([x, positions], dim=-1)  # Append position as feature
        x = nn.Linear(self.hidden_dim + 1, self.hidden_dim).to(x.device)(x)  # Project back to hidden_dim
        
        # Multi-head Attention (lighter weight)
        residual = x
        attn_output, _ = self.multihead_attention(
            query=x,
            key=x,
            value=x
        )
        x = residual + attn_output
        
        # Feed-forward block with smaller expansion
        residual = x
        x = self.feed_forward(x)
        x = residual + x
        
        # Single normalization at the end
        x = self.norm(x)
        
        # Enhance consonant features
        x = self.consonant_enhancer(x)
        x, _ = self.attention(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
class Improved_Phi_FC_Recurrent(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, num_layers=1):
        super(Improved_Phi_FC_Recurrent, self).__init__()
        
        # Keep your original mel spectrogram components
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mel_bins
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # The CNN backbone
        self.phi = MatchboxNetSkip(input_channels=n_mel_bins)  # Expects n_mel_bins channels
        
        # Use your original RNN implementation for now
        self.rnn_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = 64 if i == 0 else hidden_dim
            self.rnn_layers.append(StatefulGRU(input_dim, hidden_dim))

        # Add consonant enhancer (minimal parameters)
        self.consonant_enhancer = LightConsonantEnhancer(hidden_dim)

        # Keep your original attention
        self.attention = FocusedAttention(hidden_dim)
        
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
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
        
        # Remove the dimension we added
        x = x.squeeze(1)  # [batch, n_mels, time]
        
        # print(x.shape)  # Debugging line to check shape after normalization
        # Now x should be correctly shaped for phi: [batch, n_mels, time]
        x = self.phi(x)  # Output shape: [batch, 64, time]
        x = x.permute(0, 2, 1).contiguous()  # [batch, time, 64]
        
        # Rest of your code is the same
        h_t = None
        for rnn in self.rnn_layers:
            x, h_t = rnn(x, h_t)

        # Enhance consonant features
        x = self.consonant_enhancer(x)

        x, _ = self.attention(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
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

class Phi_FC_Recurrent(nn.Module):
    def __init__(self, num_classes=10, n_mel_bins=64, hidden_dim=32, num_layers=1):
        super(Phi_FC_Recurrent, self).__init__()

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mel_bins
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        self.phi = MatchboxNetSkip(input_channels=n_mel_bins)

        self.rnn_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = 64 if i == 0 else hidden_dim
            self.rnn_layers.append(StatefulRNNLayer(input_dim, hidden_dim))

        # self.attention = AttentionLayer(hidden_dim)
        self.attention = FocusedAttention(hidden_size=hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.mel_spec(x)
        x = self.amplitude_to_db(x)

        if x.dim() == 3:
            x = x.unsqueeze(1)

        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        x = (x - mean) / std

        x = x.squeeze(1)
        x = self.phi(x)
        x = x.permute(0, 2, 1).contiguous()  # (batch, seq_len, features)

        # Initialize hidden state
        h_t = None

        # Pass through stacked RNN layers with stateful connections
        for rnn in self.rnn_layers:
            x, h_t = rnn(x, h_t)

        x, _ = self.attention(x)
        x = self.dropout(x)
        x = self.fc2(x)
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