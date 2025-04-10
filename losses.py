import torch 
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Optional class weights
        
    def forward(self, logits, target):
        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=1)
        
        # Get the probability of the correct class for each sample
        batch_size = logits.size(0)
        p = probs[range(batch_size), target]
        
        # Apply focal loss formula: -(1-p)^gamma * log(p)
        focal_weights = (1 - p) ** self.gamma
        loss = -focal_weights * torch.log(p + 1e-10)  # Add small epsilon for numerical stability
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_weights = self.alpha[target]
            loss = alpha_weights * loss
            
        return loss.mean()