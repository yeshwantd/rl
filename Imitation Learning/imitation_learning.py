# imitation_learning.py

import os
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models import BasicPolicy # Assuming models.py is in the same directory

class ImitationDataset(Dataset):
    """
    Dataset for imitation learning from saved episodes.
    Each item is (observation, action).
    - observations: np.ndarray of shape (8,)
    - actions: int in [0, 3]
    """
    def __init__(self, path: str):
        data = np.load(path, allow_pickle=True)
        episodes = data["episodes"].tolist()  # list of episodes
        samples = []
        for ep in episodes:
            for obs, act, rew in ep:
                # obs is already a numpy array of shape (8,)
                samples.append((obs.astype(np.float32), int(act)))
        self.observations = [s[0] for s in samples]
        self.actions = [s[1] for s in samples]

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.observations[idx])
        act = torch.tensor(self.actions[idx], dtype=torch.long)
        return obs, act


def make_dataloader(path: str, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Create a DataLoader from an episodes npz file.

    Args:
        path: str, path to the .npz file (with key 'episodes')
        batch_size: int, number of samples per batch
        shuffle: bool, whether to shuffle dataset (default: True)
        num_workers: int, DataLoader workers (default: 0 for simplicity)

    Returns:
        DataLoader object
    """
    dataset = ImitationDataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Model, loss, optimizer, device

# Create model
model = BasicPolicy()

# Loss: cross-entropy for 4-way discrete action classification
loss_fn = nn.CrossEntropyLoss()

# Optimizer: AdamW (tweak lr/weight_decay as needed)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

# Device: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"[init] device: {device}")

# Training loop
def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    ckpt_dir: str,
    log_every: int = 100,
) -> Dict[str, List[float]]:
    """
    Train a policy network with cross-entropy loss on imitation data.

    Args:
        model:         nn.Module producing logits over actions.
        optimizer:     torch optimizer (e.g., AdamW).
        loss_fn:       loss function (e.g., nn.CrossEntropyLoss()).
        num_epochs:    number of training epochs.
        train_loader:  DataLoader that yields (obs, action) for training.
        test_loader:   DataLoader that yields (obs, action) for evaluation.
        ckpt_dir:      directory to save checkpoints.
        log_every:     print a minibatch log every N steps (optional).

    Returns:
        history: dict with keys:
            - "train_loss": list of per-epoch average train losses
            - "test_loss":  list of per-epoch average test losses
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    device = next(model.parameters()).device  # use model's current device
    history = {"train_loss": [], "test_loss": []}

    for epoch in range(1, num_epochs + 1):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        num_batches = 0

        for step, (obs, act) in enumerate(train_loader, start=1):
            # Move data to device
            obs = obs.to(device, non_blocking=True)
            act = act.to(device, non_blocking=True)

            # Forward
            logits = model(obs)               # [B, 4]
            loss = loss_fn(logits, act)       # scalar

            # Backprop + update
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()
            num_batches += 1

            # Optional periodic logging
            if log_every and (step % log_every == 0):
                avg_so_far = running_loss / num_batches
                print(f"[epoch {epoch:03d} | step {step:05d}] train loss (avg): {avg_so_far:.4f}")

        # Average train loss over all mini-batches
        epoch_train_loss = running_loss / max(1, num_batches)
        history["train_loss"].append(epoch_train_loss)

        # ---- Evaluation ----
        model.eval()
        test_running_loss = 0.0
        test_batches = 0
        with torch.no_grad():
            for obs, act in test_loader:
                obs = obs.to(device, non_blocking=True)
                act = act.to(device, non_blocking=True)
                logits = model(obs)
                loss = loss_fn(logits, act)
                test_running_loss += loss.item()
                test_batches += 1

        epoch_test_loss = test_running_loss / max(1, test_batches)
        history["test_loss"].append(epoch_test_loss)

        print(f"[epoch {epoch:03d}] train_loss: {epoch_train_loss:.4f} | test_loss: {epoch_test_loss:.4f}")

        # ---- Checkpoint every 5 epochs ----
        if epoch % 5 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": epoch_train_loss,
                    "test_loss": epoch_test_loss,
                },
                ckpt_path,
            )
            print(f"[ckpt] saved: {ckpt_path}")

    return history

# Create DataLoader from saved episodes
data_path = "./data/train_episodes.npz"
batch_size = 64  # you can adjust as needed

train_loader = make_dataloader(data_path, batch_size=batch_size, shuffle=True)

print(f"[init] Loaded dataset from {data_path}, batches of {batch_size}, total samples: {len(train_loader.dataset)}")

# Create DataLoader for test set
test_data_path = "./data/test_episodes.npz"
test_batch_size = 64  # can be same or different from train batch size

test_loader = make_dataloader(test_data_path, batch_size=test_batch_size, shuffle=False)

print(f"[init] Loaded test dataset from {test_data_path}, batches of {test_batch_size}, total samples: {len(test_loader.dataset)}")

# Train the model for 20 epochs, saving checkpoints to ./checkpoints
history = train(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    num_epochs=40,
    train_loader=train_loader,
    test_loader=test_loader,
    ckpt_dir="./checkpoints"
)

import matplotlib.pyplot as plt

# Plot training vs test loss curves
plt.figure(figsize=(8, 5))
plt.plot(history["train_loss"], label="Train Loss", marker="o")
plt.plot(history["test_loss"], label="Test Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Training vs Test Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()