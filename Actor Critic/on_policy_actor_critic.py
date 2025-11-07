import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from matplotlib import pyplot as plt
import gymnasium as gym

from models import BasicPolicy, BasicValue
from utils import test_policy, visualize_policy


# Generate episodic data for a given policy
def generate_episodic_data(env, policy, num_episodes, seed=None, eval_mode=False, device='cpu'):
    """
    Generate episodic data for a given policy in an environment.

    Args:
        env (gym.Env): The environment to generate data for.
        policy (callable): The policy to use for generating data.
        num_episodes (int): The number of episodes to generate.

    Returns:
        list: A list of episodes, where each episode is a list of (state, action, reward, next_state) tuples.
    """
    episodes = []
    if eval_mode:
        policy.eval()
        for param in policy.parameters():
            param.requires_grad = False

    for episode_i in range(num_episodes):
        new_seed = seed + episode_i if seed else None
        observation, info = env.reset(seed = new_seed)
        done = False
        episode = []
        while not done:
            obs_tensor = torch.tensor(observation, dtype=torch.float32).to(device)
            action_distribution = policy.get_action_distribution(obs_tensor)
            action = torch.multinomial(action_distribution, 1).item()
            action_prob = action_distribution[action]
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode.append((observation, action, reward, next_observation, action_prob))
            observation = next_observation
            done = terminated or truncated
        episodes.append(episode)
    return episodes

# Train a critic model
def train_critic(model, optimizer, episodes, gamma, device):
    """
    Train a critic model using the given episodes and optimizer.

    Args:
        model (nn.Module): The critic model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        episodes (list): A list of episodes, where each episode is a list of (observation, action, reward, next_observation) tuples.
        gamma (float): The discount factor.

    Returns:
        float: The average loss over all episodes.
    """
    losses = []
    for episode in episodes:
        loss = 0
        for i, (observation, action, reward, next_observation, action_prob) in enumerate(episode):
            observation = torch.tensor(observation, dtype=torch.float32).to(device)
            next_observation = torch.tensor(next_observation, dtype=torch.float32).to(device)
            # y_t = r(s_t,a_t) + γ * V(s_{t+1})
            target = reward + gamma * model(next_observation).detach()
            prediction = model(observation)
            loss += F.mse_loss(target, prediction)
        
        # Gradient descent on accumulated losses
        optimizer.zero_grad()
        loss = loss / len(episode)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

# Train the actor model via policy gradient
def train_actor(actor, critic, optimizer, episodes, gamma, device):
    """
    Train the actor model using the given episodes and optimizer.

    Args:
        model (nn.Module): The actor model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        episodes (list): A list of episodes, where each episode is a list of (observation, action, reward, next_observation) tuples.
        gamma (float): The discount factor.

    Returns:
        float: The average loss over all episodes.
    """
    losses = []
    for episode in episodes:
        loss = 0
        for i, (observation, action, reward, next_observation, _) in enumerate(episode):
            obs_tensor = torch.tensor(observation, dtype=torch.float32).to(device)
            next_obs_tensor = torch.tensor(next_observation, dtype=torch.float32).to(device)
            
            # Recompute action probability with current policy parameters (fixes computational graph issue)
            action_distribution = actor.get_action_distribution(obs_tensor)
            action_prob = action_distribution[action]
            
            advantage = reward + gamma * critic(next_obs_tensor) - critic(obs_tensor)
            loss += -torch.log(action_prob) * advantage
        
        # Gradient descent on accumulated losses over episode
        optimizer.zero_grad()
        loss = loss / len(episode)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


# Train the actor and critic models
def main():
    
    SHOW_PLOTS = False
    
    num_epochs = 4000
    num_critic_warm_start_epochs = 100
    num_episodes = 10
    gamma = 0.99
    seed = 42
    max_episode_steps = 500
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("LunarLander-v3", max_episode_steps=max_episode_steps)
    actor = BasicPolicy().to(device)
    critic = BasicValue().to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-1)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    
    critic_losses = []
    # Warm start by training the critic network on sum of discounted rewards
    for i in range(num_critic_warm_start_epochs):
        episodes = generate_episodic_data(env, actor, num_episodes, seed=seed, eval_mode=True, device=device)
        critic_loss = train_critic(critic, critic_optimizer, episodes, gamma, device)
        critic_losses.append(critic_loss)
        print(f"Warm start epoch: {i}, Critic Loss: {critic_loss}")
    
    # Plot Critic loss
    if SHOW_PLOTS:
        plt.plot(critic_losses, label="Critic Loss")
        plt.show()

    # Use the warm started critic network to train the actor network
    actor_losses, critic_losses = [], []
    best_rewards = -np.inf
    for epoch in range(num_epochs):
        episodes = generate_episodic_data(env, actor, num_episodes, seed=seed, device=device)
        critic_loss = train_critic(critic, critic_optimizer, episodes, gamma, device)
        actor_loss = train_actor(actor, critic, actor_optimizer, episodes, gamma, device)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        print(f"Epoch: {epoch}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")

        # Test the goodness of the current policy for every 10 epochs
        if (epoch + 1)%10 == 0:
            avg_rewards = np.mean(test_policy(env, actor))
            print(f"Average rewards for policy after {epoch+1} epochs: {avg_rewards}")
            
            # Save checkpoint if rewards better than previous best rewards
            if avg_rewards > best_rewards:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(actor.state_dict(), "checkpoints/best_policy.pt")
        
    if SHOW_PLOTS:
        plt.plot(actor_losses, label="Actor Loss")
        plt.plot(critic_losses, label="Critic Loss")
        plt.legend()
        plt.show()

    # Visualize the best policy
    actor.load_state_dict(torch.load("checkpoints/best_policy.pt"))
    env = gym.make("LunarLander-v3", render_mode="human")
    visualize_policy(env, actor)

if __name__ == "__main__":
    main()
    

    
