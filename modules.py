from torch import nn
import torch.nn.functional as F
import torch

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, gru_output):
        # gru_output shape: (batch, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(gru_output), dim=1)
        context_vector = torch.sum(attention_weights * gru_output, dim=1)
        return context_vector, attention_weights