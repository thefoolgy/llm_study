import torch
import torch.nn as nn

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    log_sum_exp = x_max + torch.log(torch.exp(x - x_max).sum(dim=dim, keepdim=True))
    return x - log_sum_exp

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, targets):
        log_probs = log_softmax(logits)
        batch_size = logits.size(0)
        log_probs_for_targets = log_probs[range(batch_size), targets]
        loss = -log_probs_for_targets
        return loss.mean()
    
    def calculate_perplexity(self, logits, targets):
        return torch.exp(self.forward(logits, targets))