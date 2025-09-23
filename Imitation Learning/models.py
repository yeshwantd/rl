# models.py
import torch
import torch.nn as nn

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