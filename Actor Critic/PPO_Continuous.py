import torch
from torch.nn import Module, Linear, ReLU, Sequential, Dropout, Parameter
import torch.nn.functional as F
from torch.distributions import Normal

from torch.optim import Adam
import numpy as np
import gymnasium as gym
import copy
import random
from collections import deque
import os
import sys
import time

import pickle

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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
        self.fc3 = Linear(256, 2) # Two outputs each for mean and log_std
        self.log_std = Parameter(torch.zeros(2)) # state independent standard deviation

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x)) # Actions are in [-1, 1]
        std = self.log_std.exp()
        if x.dim() > 1:
            std = std.unsqueeze(0).expand(x.size(0), -1) # Expand std to match the batch size
        return mean, std

def train():
    # Configs
    epochs = 2000
    num_test_epochs = 100
    ppo_epochs = 10
    batch_size = 16
    num_test_runs = 10
    max_episode_steps = 500

    clip_param = 0.2
    gamma = 0.99
    policy_lr = 3e-4
    value_lr = 3e-4
    adam_eps = 1e-5
    best_test_reward = -np.inf
    train_seed = 42
    test_seed = 55
    rewards_scale = 0.01

    normalize_obs = False if "--no-norm" in sys.argv else True
    set_seed = False if "--no-seed" in sys.argv else True
    enable_gae = False if "--no-gae" in sys.argv else True
    lr_anneal = False if "--no-anneal" in sys.argv else True

    # Reproducibility
    if set_seed and train_seed is not None:
        torch.manual_seed(train_seed)
        np.random.seed(train_seed)
        random.seed(train_seed)
        
    # Initialize environment
    env = gym.make("LunarLanderContinuous-v3", max_episode_steps=max_episode_steps)
    if normalize_obs:
        obs_normalizer = RunningMeanStd(shape=(8,))

    # Initialize actor and critic
    policy = PolicyNetwork()
    prev_policy = copy.deepcopy(policy)
    value = ValueNetwork()

    # Disable gradients for previous policy 
    for param in prev_policy.parameters():
        param.requires_grad = False

    # Optimizers
    policy_optim = Adam(policy.parameters(), lr=policy_lr, eps=adam_eps)
    value_optim = Adam(value.parameters(), lr=value_lr, eps=adam_eps)

    # Train loop
    for epoch in range(epochs):
        # Annealing the rate if instructed to do so.
        if lr_anneal:
            frac = 1.0 - (epoch - 1.0) / epochs
            policy_optim.param_groups[0]["lr"] = policy_lr * frac
            value_optim.param_groups[0]["lr"] = value_lr * frac

        # Collect set of trajectories
        observations, actions, rewards, next_observations, dones, action_log_probs = [], [], [], [], [], []
        for i in range(batch_size):
            seed = train_seed + epoch if set_seed and train_seed is not None else None
            obs, info = env.reset(seed=seed)
            if normalize_obs:
                obs = obs_normalizer.normalize(obs)
            done = False
            while not done:
                mean, std = prev_policy(torch.tensor(obs, dtype=torch.float32))
                dist = Normal(mean, std)
                action = dist.sample()
                action_log_prob = dist.log_prob(action).sum(dim=-1) # There are 2 actions, independent of each other. So, joint log prob is sum of individual log probs.
                entropy = dist.entropy().sum(dim=-1) # Joint entropy is sum of individual entropies.
                next_obs, reward, terminated, truncated, info = env.step(action.numpy())
                
                # Update normalizer with raw next_obs, then normalize it
                if normalize_obs:
                    obs_normalizer.update(np.array([next_obs]))
                    next_obs = obs_normalizer.normalize(next_obs)
                
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

        # Compute Advantages
        with torch.no_grad():
            values = value(observations).squeeze()
            next_values = value(next_observations).squeeze()
            
            if enable_gae:
                deltas = rewards * rewards_scale + gamma * (1.0 - dones) * next_values - values
                advantages = torch.zeros_like(deltas)
                gae = 0
                gae_lambda = 0.95
                for t in reversed(range(len(rewards))):
                    gae = deltas[t] + gamma * gae_lambda * (1.0 - dones[t]) * gae
                    advantages[t] = gae
                target_values = advantages + values
            else:
                # Simple 1-step advantage
                target_values = rewards * rewards_scale + gamma * (1.0 - dones) * next_values
                advantages = target_values - values
            
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update policy with Mini-batches
        dataset_size = len(observations)
        indices = np.arange(dataset_size)
        minibatch_size = 64

        policy.train()
        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, minibatch_size):
                end = start + minibatch_size
                idx = indices[start:end]
                
                mb_obs = observations[idx]
                mb_actions = actions[idx]
                mb_advantages = advantages[idx]
                mb_old_log_probs = action_log_probs[idx]
                mb_target_values = target_values[idx]

                action_means, action_std = policy(mb_obs)
                dist = Normal(action_means, action_std)
                entropy = dist.entropy().sum(dim=-1).mean()
                new_action_log_probs = dist.log_prob(mb_actions).sum(dim=-1)

                # Compute PPO loss
                ratio = torch.exp(new_action_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean() # PPO loss
                policy_loss -= 0.01 * entropy # Entropy regularization

                # Update policy
                policy_optim.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                policy_optim.step()

                # Update value
                value_optim.zero_grad()
                value_loss = F.mse_loss(value(mb_obs).squeeze(), mb_target_values.squeeze())
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
                seed = test_seed + i if set_seed and test_seed is not None else None
                obs, info = env.reset(seed=seed)
                if normalize_obs:
                    obs = obs_normalizer.normalize(obs)
                done = False
                policy.eval()
                while not done:
                    with torch.no_grad():
                        mean, std = policy(torch.tensor(obs, dtype=torch.float32))
                        dist = Normal(mean, std)
                        action = dist.sample()
                    next_obs, reward, terminated, truncated, info = env.step(action.numpy())
                    if normalize_obs:
                        next_obs = obs_normalizer.normalize(next_obs)
                    done = terminated or truncated
                    episode_reward += reward
                    obs = next_obs
                test_rewards.append(episode_reward)
            print(f"Epoch {epoch + 1}, Avg Test Reward: {np.mean(test_rewards)}")

            # Save best policy
            if np.mean(test_rewards) > best_test_reward:
                best_test_reward = np.mean(test_rewards)
                script_dir = os.path.dirname(os.path.abspath(__file__))
                checkpoint_dir = os.path.join(script_dir, "checkpoints")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                torch.save(policy.state_dict(), os.path.join(checkpoint_dir, "ppo_continuous_policy.pth"))
                if normalize_obs:
                    with open(os.path.join(checkpoint_dir, "obs_normalizer.pkl"), "wb") as f:
                        pickle.dump(obs_normalizer, f)

    # Close environment
    env.close()
    
def render(policy, obs_normalizer, num_times):
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")
    for i in range(num_times):    
        obs, info = env.reset()
        if obs_normalizer is not None:
            obs = obs_normalizer.normalize(obs)
        done = False
        tot_reward = 0
        policy.eval()
        with torch.no_grad():
            while not done:
                mean, _ = policy(torch.tensor(obs, dtype=torch.float32))
                action = mean.numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                if obs_normalizer is not None:
                    obs = obs_normalizer.normalize(obs)
                tot_reward += reward
                done = terminated or truncated
            print(f"Total reward for episode {i+1}: {tot_reward}")
    env.close()

if __name__ == "__main__":
    train_flag = False
    render_flag = True
    if train_flag:
        start_time = time.time()
        train()
        end_time = time.time()
        print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")
    if render_flag:
        policy = PolicyNetwork()
        policy.load_state_dict(torch.load("checkpoints/ppo_continuous_policy.pth"))
        obs_normalizer = None
        if os.path.exists("checkpoints/obs_normalizer.pkl"):
            with open("checkpoints/obs_normalizer.pkl", "rb") as f:
                obs_normalizer = pickle.load(f)
                
        render(policy, obs_normalizer, 5)