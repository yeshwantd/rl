# Soft Actor-Critic (SAC)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import copy
import numpy as np
import gymnasium as gym
from collections import deque
import random
import time
import os
import sys
import pickle

class RunningMeanStd:
    """
    Tracks the running mean and variance of a data stream.
    Used for observation normalization.
    """
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x):
        """
        Updates the running mean and variance with a new batch of data.
        """
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
        """
        Normalizes the input using the running mean and standard deviation.
        """
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

class PolicyNetwork(nn.Module):
    """
    Parametrized policy πθ(a|s).
    Uses a Gaussian distribution for continuous actions.
    """
    def __init__(self, obs_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim) 
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, x):
        """
        Computes the mean and log-std of the Gaussian distribution.
        """
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        mean = self.mean(x) 
        log_std = self.log_std(x)
        # Clamp log_std to [min_val, max_val] for numerical stability
        log_std = torch.clamp(log_std, min=-20, max=2) 
        return mean, log_std

    def sample(self, x):
        """
        Samples an action using the reparameterization trick and applies tanh squashing.
        Returns:
            action: Squashed action in [-1, 1]
            log_prob: Corrected log-probability of the action
        """
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Reparameterization trick: u = mean + std * epsilon, where epsilon ~ N(0, 1)
        u = normal.rsample()  
        action = torch.tanh(u) # Enforce action bounds from -1 to 1

        # Calculate log probability of the action
        log_prob = normal.log_prob(u)
        # Apply Jacobian correction for the tanh transformation
        # log π(a|s) = log µ(u|s) - sum(log(1 - tanh^2(u)))
        log_prob -= torch.log(1 - action.pow(2) + 1e-6) 
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob
        
class QNetwork(nn.Module):
    """
    Soft Q-function Qφ(s, a).
    Computes the expected return of taking action 'a' in state 's'.
    """
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        """
        x is a concatenation of [observation, action]
        """
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
def train(checkpoint_path):
    # Training and test configs
    epochs = 2000
    test_epochs_freq = 100
    steps_per_epoch = 256
    batch_size = 256
    num_test_episodes = 10
    num_trajectories = 10
    best_test_reward = 5000

    # Reproducibility
    train_seed = 42
    test_seed = 55
    set_seed = False if "--no-seed" in sys.argv else True
    if set_seed and train_seed is not None:
        torch.manual_seed(train_seed)
        np.random.seed(train_seed)
        random.seed(train_seed)

    # Environment configs
    env_name = "Humanoid-v5"
    max_episode_steps = 1000
    
    # Agent configs
    gamma = 0.99
    policy_lr = 3e-4
    q_lr = 3e-4
    alpha_lr = 3e-4
    reward_scale = 1.0
    
    # Replay buffer configs
    replay_buffer_size = 100000
    
    # Target network configs
    tau = 0.005


    # Initialize the environment
    env = gym.make(env_name, max_episode_steps=max_episode_steps)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Action space bounds for scaling
    action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
    action_high = torch.tensor(env.action_space.high, dtype=torch.float32)
    
    obs_normalizer = RunningMeanStd(shape=(obs_dim,))
    
    # Initialize the actor and critic networks
    policy = PolicyNetwork(obs_dim, action_dim)
    q1 = QNetwork(obs_dim, action_dim)
    q2 = QNetwork(obs_dim, action_dim)
    
    # Initialize the target networks and set parameters equal to the original networks
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)
    
    # Initialize the optimizers
    policy_optim = optim.Adam(policy.parameters(), lr=policy_lr)
    q1_optim = optim.Adam(q1.parameters(), lr=q_lr)
    q2_optim = optim.Adam(q2.parameters(), lr=q_lr)

    # Automatic entropy tuning
    target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(torch.device("cpu"))).item()
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_optim = optim.Adam([log_alpha], lr=alpha_lr)
    alpha = log_alpha.exp()
    
    # Initialize the replay buffer
    replay_buffer = deque(maxlen=replay_buffer_size)
    
    # Training loop
    for epoch in range(epochs):
        # Collect a batch of trajectories
        for i in range(num_trajectories):
            seed = train_seed + epoch if set_seed and train_seed is not None else None
            obs, info = env.reset(seed=seed)
            obs = obs_normalizer.normalize(obs)
            done = False
            while not done:
                # Sample an action from the policy (in [-1, 1] range)
                with torch.no_grad():
                    action, _ = policy.sample(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = action.squeeze(0).cpu().numpy()
                
                # Scale action to environment bounds: [-1, 1] -> [low, high]
                env_action = env.action_space.low + (action + 1.0) * 0.5 * (env.action_space.high - env.action_space.low)
                
                next_obs, reward, terminated, truncated, info = env.step(env_action)
                
                # Update normalizer with raw next_obs, then normalize it
                obs_normalizer.update(np.array([next_obs]))
                next_obs = obs_normalizer.normalize(next_obs)
                
                done = terminated or truncated
                # Store unscaled action in replay buffer for consistent network logic
                replay_buffer.append((obs, action, reward * reward_scale, next_obs, done))
                obs = next_obs
        
        # Sample a batch of steps from the replay buffer
        for i in range(steps_per_epoch):
            if len(replay_buffer) < batch_size:
                break
            batch = random.sample(replay_buffer, batch_size)
            b_obs, b_action, b_reward, b_next_obs, b_done = zip(*batch)
            
            # Convert to tensors
            b_obs = torch.tensor(np.array(b_obs), dtype=torch.float32)
            b_action = torch.tensor(np.array(b_action), dtype=torch.float32)
            b_reward = torch.tensor(np.array(b_reward), dtype=torch.float32).unsqueeze(1)
            b_next_obs = torch.tensor(np.array(b_next_obs), dtype=torch.float32)
            b_done = torch.tensor(np.array(b_done), dtype=torch.int8).unsqueeze(1)
            
            # Compute targets for Q-functions
            with torch.no_grad():
                # Sample next actions and their log-probabilities from current policy
                next_action, next_log_prob = policy.sample(b_next_obs)
                next_state_action = torch.cat([b_next_obs, next_action], dim=1)
                
                # Clipped Double-Q trick: use the minimum of two target Q-networks
                # This reduces overestimation bias in Q-learning.
                q1_next = q1_target(next_state_action)
                q2_next = q2_target(next_state_action)
                min_q_next = torch.min(q1_next, q2_next)
                
                # Bellman equation with entropy term:
                # y = r + γ * (1 - d) * (Q_target(s', a') - α * log_π(a'|s'))
                y = b_reward + gamma * (1 - b_done) * (min_q_next - alpha * next_log_prob)

            # Update Q-functions
            state_action = torch.cat([b_obs, b_action], dim=1)
            q1_loss = F.mse_loss(q1(state_action), y)  # J(φ) = E [1/2 * (Qφ(s,a) - y)^2]
            q2_loss = F.mse_loss(q2(state_action), y)
            
            q1_optim.zero_grad()
            q1_loss.backward()
            q1_optim.step()

            q2_optim.zero_grad()
            q2_loss.backward()
            q2_optim.step()

            # Update Policy
            # Sample current actions using current policy
            new_action, log_prob = policy.sample(b_obs)
            q1_new = q1(torch.cat([b_obs, new_action], dim=1))
            q2_new = q2(torch.cat([b_obs, new_action], dim=1))
            # Use the minimum of current Q-networks for the policy objective
            min_q_new = torch.min(q1_new, q2_new)
            
            # Policy objective: Maximize (Q - α * log_π)
            # We minimize -J(θ) = -E [Qφ(s, a) - α * log(πθ(a|s))]
            policy_loss = -(min_q_new - alpha * log_prob).mean()
            
            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

            # Update Alpha (entropy weight)
            alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()
            alpha = log_alpha.exp()

            # Soft update target networks using Exponential Moving Average (EMA)
            # Q_target = τ * Q_online + (1 - τ) * Q_target
            for target_param, param in zip(q1_target.parameters(), q1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            for target_param, param in zip(q2_target.parameters(), q2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        # Test the policy
        if (epoch + 1) % test_epochs_freq == 0:
            test_rewards = []
            for i in range(num_test_episodes):
                obs, info = env.reset(seed = test_seed + i if set_seed and test_seed is not None else None)
                obs = obs_normalizer.normalize(obs)
                done = False
                episode_reward = 0
                while not done:
                    with torch.no_grad():
                        action, _ = policy.sample(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    action = action.squeeze(0).cpu().numpy()
                    
                    # Scale action to environment bounds: [-1, 1] -> [low, high]
                    env_action = env.action_space.low + (action + 1.0) * 0.5 * (env.action_space.high - env.action_space.low)
                    
                    obs, reward, terminated, truncated, info = env.step(env_action)
                    obs = obs_normalizer.normalize(obs)
                    done = terminated or truncated
                    episode_reward += reward
                if episode_reward > best_test_reward:
                    best_test_reward = episode_reward
                    torch.save(policy.state_dict(), checkpoint_path)
                    with open("checkpoints/sac_humanoid_obs_normalizer.pkl", "wb") as f:
                        pickle.dump(obs_normalizer, f)
                test_rewards.append(episode_reward)
            print(f"Epoch {epoch+1}, Avg Test Reward: {np.mean(test_rewards)}")

    env.close()

# Demo the policy
def demo(policy, obs_normalizer, num_times):
    env = gym.make("Humanoid-v5", render_mode="human")
    for i in range(num_times):    
        obs, info = env.reset()
        if obs_normalizer is not None:
            obs = obs_normalizer.normalize(obs)
        done = False
        tot_reward = 0
        policy.eval()
        with torch.no_grad():
            while not done:
                action, _ = policy.sample(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = action.squeeze(0).cpu().numpy()
                
                # Scale action to environment bounds: [-1, 1] -> [low, high]
                env_action = env.action_space.low + (action + 1.0) * 0.5 * (env.action_space.high - env.action_space.low)
                
                obs, reward, terminated, truncated, info = env.step(env_action)
                if obs_normalizer is not None:
                    obs = obs_normalizer.normalize(obs)
                tot_reward += reward
                done = terminated or truncated
            print(f"Total reward for episode {i+1}: {tot_reward}")
    env.close()

if __name__ == "__main__":
    train_flag = False  
    render_flag = True
    checkpoint_path = "checkpoints/sac_humanoid_policy.pth"
    if train_flag:
        start_time = time.time()
        train(checkpoint_path)
        end_time = time.time()
        print(f"Total Training Time: {(end_time - start_time)/60:.2f} minutes")
    elif render_flag:
        env = gym.make("Humanoid-v5", render_mode="human")
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()
        
        policy = PolicyNetwork(obs_dim, action_dim)
        policy.load_state_dict(torch.load(checkpoint_path))
        
        obs_normalizer = None
        if os.path.exists("checkpoints/sac_humanoid_obs_normalizer.pkl"):
            with open("checkpoints/sac_humanoid_obs_normalizer.pkl", "rb") as f:
                obs_normalizer = pickle.load(f)
                
        demo(policy, obs_normalizer, 5)