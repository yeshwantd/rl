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
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt

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
        
        self.observations = []
        self.actions = []
        
        for ep in episodes:
            for obs, act in ep:
                self.observations.append(torch.from_numpy(obs.astype(np.float32)))
                self.actions.append(torch.tensor(act, dtype=torch.long))

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


def make_dataloader(dataset_path: str, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
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
    dataset = ImitationDataset(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Training loop
def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    ckpt_dir: str,
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
        
    Returns:
        history: dict with keys:
            - "train_loss": list of per-epoch average train losses
            - "test_loss":  list of per-epoch average test losses
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    device = next(model.parameters()).device  # use model's current device
    history = {"train_loss": [], "test_loss": []}
    
    # Track best model
    best_test_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        
        # ================ Training ================
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

        # Average train loss over all mini-batches
        epoch_train_loss = running_loss / max(1, num_batches)
        history["train_loss"].append(epoch_train_loss)

        # ================ Evaluation ================
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
        
        # Print train and test loss
        print(f"Epoch {epoch+1}: train_loss: {epoch_train_loss:.4f}, test_loss: {epoch_test_loss:.4f}")
        
        #  ================ Save best model  ================
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            best_epoch = epoch
            best_model_path = os.path.join(ckpt_dir, "best_policy.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": epoch_train_loss,
                    "test_loss": epoch_test_loss,
                },
                best_model_path,
            )
    
    return history

def plot_training_curves(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training and test loss curves.
    
    Args:
        history: Dictionary with 'train_loss' and 'test_loss' keys
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o", linewidth=2)
    plt.plot(epochs, history["test_loss"], label="Test Loss", marker="s", linewidth=2)
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average Loss", fontsize=12)
    plt.title("Imitation Learning: Training vs Test Loss", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def visualize_policy(model: nn.Module, env_name: str = "LunarLander-v3", 
                    num_episodes: int = 5, max_steps: int = 1000, device: str = 'cpu'):
    """
    Visualize the trained policy in the environment with human rendering.
    
    Args:
        model: Trained policy model
        env_name: Name of the Gymnasium environment
        num_episodes: Number of episodes to visualize
        max_steps: Maximum steps per episode
        device: Device the model is on
    """
    import gymnasium as gym
        
    env = gym.make(env_name, render_mode='human', max_episode_steps=max_steps)
    model.eval()
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        with torch.no_grad():
            while not done and step_count < max_steps:
                # Get action from policy
                obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(obs_tensor)
                action = torch.argmax(logits, dim=1).item()
                
                # Take action
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        print(f"Episode {episode + 1} Reward: {episode_reward:.2f} Steps: {step_count}")
    env.close()

def main():
    """Main training and evaluation pipeline."""
    
    # Configuration
    TRAIN_DATA_PATH = "./data/training_data.npz"
    TEST_DATA_PATH = "./data/test_data.npz"
    CHECKPOINT_DIR = "./checkpoints"
    PLOT_SAVE_PATH = "./checkpoints/training_curves.png"
    
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-2
    
    VISUALIZE_POLICY = True
    NUM_VIS_EPISODES = 3
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] Using device: {device}")
    
    # Create model
    model = BasicPolicy()
    model = model.to(device)
    
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Load data
    train_loader = make_dataloader(TRAIN_DATA_PATH, batch_size=BATCH_SIZE, shuffle=True)    
    test_loader = make_dataloader(TEST_DATA_PATH, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train the model
    history = train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=NUM_EPOCHS,
        train_loader=train_loader,
        test_loader=test_loader,
        ckpt_dir=CHECKPOINT_DIR,
    )
        
    
    # Plot training curves
    plot_training_curves(history, save_path=PLOT_SAVE_PATH)
    
    # Load best model for visualization
    if VISUALIZE_POLICY:        
        best_model_path = os.path.join(CHECKPOINT_DIR, "best_policy.pt")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print(f"Warning: Best model not found at {best_model_path}")
        
        visualize_policy(model, num_episodes=NUM_VIS_EPISODES, device=device)

if __name__ == "__main__":
    main()