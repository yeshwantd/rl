# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicPolicy(nn.Module):
    """
    Feedforward policy:
      8 -> 16 -> 32 -> 64 -> 32 -> 16 -> 4 (tanh in between, logits out)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh(),
            nn.Linear(16, 32), nn.Tanh(),
            nn.Linear(32, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 4)   # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_action_distribution(self, x: torch.Tensor):
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def get_action(self, x: torch.Tensor, deterministic=True):
        action_distribution = self.get_action_distribution(x)
        if deterministic:
            return torch.argmax(action_distribution).item()
        else:
            action = torch.multinomial(action_distribution, num_samples=1)
        return action.item()

class BasicPolicyWithLayerNorm(BasicPolicy):
    """
    Feedforward policy with layer normalization:
      8 -> 16 -> 32 -> 64 -> 32 -> 16 -> 4 (tanh in between, logits out)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16), nn.LayerNorm(16), nn.Tanh(),
            nn.Linear(16, 32), nn.LayerNorm(32), nn.Tanh(),
            nn.Linear(32, 64), nn.LayerNorm(64), nn.Tanh(),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.Tanh(),
            nn.Linear(32, 16), nn.LayerNorm(16), nn.Tanh(),
            nn.Linear(16, 4)   # logits
        )