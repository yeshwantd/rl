import torch
from torch.nn import Module, Linear, ReLU, Sequential, Dropout
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
import copy
import random
from collections import deque
import os

class ValueNetwork(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(8, 256)
        self.fc2 = Linear(256, 256)
        self.fc3 = Linear(256, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyNetwork(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(8, 256)
        self.fc2 = Linear(256, 256)   
        self.fc3 = Linear(256, 4)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

def train():
    # Configs
    epochs = 1000
    num_test_epochs = 100
    ppo_epochs = 10
    batch_size = 64
    clip_param = 0.2
    gamma = 0.99
    best_test_reward = -np.inf
    train_seed = 42
    test_seed = 55
    num_test_runs = 10

    # Reproducibility
    if train_seed is not None:
        torch.manual_seed(train_seed)
        np.random.seed(train_seed)
        random.seed(train_seed)
        
    # Initialize environment
    env = gym.make("LunarLander-v3")

    # Initialize actor and critic
    policy = PolicyNetwork()
    prev_policy = copy.deepcopy(policy)
    value = ValueNetwork()

    # Disable gradients for previous policy 
    for param in prev_policy.parameters():
        param.requires_grad = False

    # Optimizers
    policy_optim = Adam(policy.parameters(), lr=1e-4)
    value_optim = Adam(value.parameters(), lr=1e-4)

    # Train loop
    for epoch in range(epochs):

        # Collect set of trajectories
        observations, actions, rewards, next_observations, dones, action_log_probs = [], [], [], [], [], []
        for i in range(batch_size):
            obs, info = env.reset(seed=train_seed + epoch if train_seed is not None else None)
            done = False
            while not done:
                logits = prev_policy(torch.tensor(obs, dtype=torch.float32))
                dist = Categorical(logits=logits)
                action = dist.sample()
                action_log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                next_obs, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                next_observations.append(next_obs)
                dones.append(done)
                action_log_probs.append(action_log_prob)
                obs = next_obs
        
        # Convert to tensors
        observations = torch.tensor(np.array(observations), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        action_log_probs = torch.tensor(np.array(action_log_probs), dtype=torch.float32)

        # Compute advantages using value network
        with torch.no_grad():
            advantages = rewards + gamma * (1.0 - dones) * value(next_observations).squeeze() - value(observations).squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # target values for value network
        with torch  .no_grad():
            target_values = rewards + gamma * (1.0 - dones) * value(next_observations).squeeze()

        # Update policy
        policy.train()
        for _ in range(ppo_epochs):
            action_logits = policy(observations)
            dist = Categorical(logits=action_logits)
            entropy = dist.entropy().mean()
            new_action_log_probs = dist.log_prob(actions)

            # Compute PPO loss
            ratio = torch.exp(new_action_log_probs - action_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() # PPO loss
            policy_loss -= 0.01 * entropy # Entropy regularization

            # Update policy
            policy_optim.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            policy_optim.step()

            # Update value
            value_optim.zero_grad()
            value_loss = F.mse_loss(value(observations).squeeze(), target_values)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), max_norm=0.5)
            value_optim.step()

        # Update previous policies weights
        for prev_param, param in zip(prev_policy.parameters(), policy.parameters()):
            prev_param.data.copy_(param.data)
    
        # Test policy every num_test_epochs epochs
        if (epoch + 1) % num_test_epochs == 0:
            test_rewards = []
            for i in range(num_test_runs):
                episode_reward = 0
                obs, info = env.reset(seed=test_seed + i if test_seed is not None else None)
                done = False
                policy.eval()
                while not done:
                    with torch.no_grad():
                        logits = policy(torch.tensor(obs, dtype=torch.float32))
                        dist = Categorical(logits=logits)
                        action = dist.sample()
                    next_obs, reward, terminated, truncated, info = env.step(action.item())
                    done = terminated or truncated
                    episode_reward += reward
                test_rewards.append(episode_reward)
            print(f"Epoch {epoch + 1}, Avg Test Reward: {np.mean(test_rewards)}")

            # Save best policy
            if np.mean(test_rewards) > best_test_reward:
                best_test_reward = np.mean(test_rewards)
                script_dir = os.path.dirname(os.path.abspath(__file__))
                checkpoint_dir = os.path.join(script_dir, "checkpoints")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                torch.save(policy.state_dict(), os.path.join(checkpoint_dir, "ppo_policy.pth"))

    # Close environment
    env.close()
    
if __name__ == "__main__":
    train()