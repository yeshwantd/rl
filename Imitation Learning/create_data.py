"""
Create Training Data for Imitation Learning

This script:
1. Loads a trained policy checkpoint (good_policy.pt)
2. Runs the policy to collect (observation, action) pairs
3. Saves the data as a .npz file for imitation learning

Usage:
    python create_training_data.py
"""

import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Policy Gradients'))

from models import BasicPolicy


def load_policy(checkpoint_path: str, device: str = 'cpu') -> BasicPolicy:
    """
    Load a trained policy from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Loaded BasicPolicy model
    """
    policy = BasicPolicy()
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    
    policy.eval()  # Set to evaluation mode
    policy.to(device)
    
    return policy


def collect_episode_data(env, policy, device: str = 'cpu', seed: int = None):
    """
    Collect (observation, action) tuples for a single episode.
    
    Args:
        env: Gymnasium environment
        policy: Trained policy
        device: Device the policy is on
        seed: Random seed for episode (optional)
    
    Returns:
        List of (observation, action) tuples
    """
    episode_data = []
    
    observation, info = env.reset(seed=seed)
    done = False
    
    with torch.no_grad():
        while not done:
            # Convert observation to tensor
            obs_tensor = torch.tensor(observation, dtype=torch.float32).to(device)
            
            # Get action from policy (deterministic for expert demonstrations)
            action = policy.get_action(obs_tensor, deterministic=True)
            
            # Take action in environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            # Store (observation, action, reward) tuple
            episode_data.append((observation.copy(), action))
            
            observation = next_observation
            done = terminated or truncated
    
    return episode_data


def collect_training_data(
    policy_path: str,
    num_episodes: int = 100,
    max_episode_steps: int = 1000,
    seed: int = None,
    device: str = 'cpu'
):
    """
    Collect training data from a trained policy.
    
    Args:
        policy_path: Path to policy checkpoint
        num_episodes: Number of episodes to collect
        max_episode_steps: Maximum steps per episode
        seed: Random seed for reproducibility
        device: Device to run policy on
    
    Returns:
        Tuple of (all_episodes, episode_rewards, episode_lengths) where:
        - all_episodes: List of episodes, each episode is a list of (observation, action, reward) tuples
        - episode_rewards: List of total rewards for each episode
        - episode_lengths: List of lengths for each episode
    """
    # Load policy
    policy = load_policy(policy_path, device=device)
    
    # Create environment
    env = gym.make("LunarLander-v3", max_episode_steps=max_episode_steps)
    
    # Collect data
    all_episodes = []
    
    for episode in range(num_episodes):
        # Use different seed for each episode if seed is provided
        episode_seed = (seed + episode) if seed is not None else None
        
        # Collect episode data
        episode_data = collect_episode_data(env, policy, device=device, seed=episode_seed)
        
        all_episodes.append(episode_data)
        
    env.close()
    
    return all_episodes


def save_training_data(
    episodes,
    output_path: str,
):
    """
    Save collected episodes to .npz file.
    
    Args:
        episodes: List of episodes (each episode is list of (obs, action, reward) tuples)
        output_path: Path to save the .npz file
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save episodes directly as numpy array
    np.savez(output_path, episodes=np.array(episodes, dtype=object))


def main():
    """
    Main function to create training and test data.
    
    Data format:
    - Each episode contains a list of (observation, action, reward) tuples
    - observation: numpy array of shape (8,) for LunarLander
    - action: integer (0-3) representing the action taken
    - reward: float representing the reward received for that step
    """
    
    # Configuration
    POLICY_PATH = "trained_policy/good_policy.pt"
    TRAIN_OUTPUT_PATH = "data/training_data.npz"
    TEST_OUTPUT_PATH = "data/test_data.npz"
    NUM_TRAIN_EPISODES = 100
    NUM_TEST_EPISODES = 20
    ENV_NAME = "LunarLander-v3"
    MAX_EPISODE_STEPS = 500
    TRAIN_SEED = 42  # Fixed seed for training data reproducibility
    TEST_SEED = 84  # Different seed for test data
        
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # Collect training data
    train_episodes = collect_training_data(
        policy_path=POLICY_PATH,
        num_episodes=NUM_TRAIN_EPISODES,
        max_episode_steps=MAX_EPISODE_STEPS,
        seed=TRAIN_SEED,
        device=device
    )
    
    # Save training data
    save_training_data(
        episodes=train_episodes,
        output_path=TRAIN_OUTPUT_PATH
    )
    
    # Collect test data
    test_episodes = collect_training_data(
        policy_path=POLICY_PATH,
        num_episodes=NUM_TEST_EPISODES,
        max_episode_steps=MAX_EPISODE_STEPS,
        seed=TEST_SEED,
        device=device
    )
    
    # Save test data
    save_training_data(
        episodes=test_episodes,
        output_path=TEST_OUTPUT_PATH
    )
    
if __name__ == "__main__":
    main()
