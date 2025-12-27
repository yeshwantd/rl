import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import matplotlib.pyplot as plt
import random
import os, time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_envs(num_envs):
    def make_env():
        def thunk():
            env = gym.make("LunarLander-v3")
            return env
        return thunk
    # envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    envs = SyncVectorEnv([make_env() for _ in range(num_envs)])
    return envs

# Policy
class Policy(Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(8, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 4)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Value function
class Value(Module):
    def __init__(self):
        super(Value, self).__init__()
        self.fc1 = torch.nn.Linear(8, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

def compute_rewards_to_go(rewards, gamma=0.99):
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
    T = rewards.shape[0]
    discounts = gamma ** torch.arange(T)
    x = rewards * discounts
    y = torch.flip(torch.cumsum(torch.flip(x, dims=[0]), dim=0), dims=[0])
    return y/discounts

# Train
def train():
    # Variables
    num_epochs = 2000
    num_episodes_per_epoch = 64
    num_envs = 64
    test_every = 100 # tests every n epochs
    seed = 42
    test_seed = 55
    num_test_episodes = 10
    best_reward = 0
    entropy_coef_start = 0.02
    entropy_coef_end = 0.001
    gamma = 0.99

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize
    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    value = Value().to(device)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=3e-4)
    envs = make_envs(num_envs)
    
    # Train
    for epoch in range(num_epochs):
        policy.train()
        observations = [[] for _ in range(num_envs)]
        next_observations = [[] for _ in range(num_envs)]
        actions = [[] for _ in range(num_envs)]
        rewards  = [[] for _ in range(num_envs)]
        discounted_sum_of_rewards = [[] for _ in range(num_envs)]
        
        # Collect data - disable autograd
        with torch.no_grad():
            seeds = [seed + epoch * num_envs + i for i in range(num_envs)] if seed else None
            obs, info = envs.reset(seed = seeds)
            done = np.zeros(num_envs, dtype=bool)    
            while not np.all(done):
                logits = policy(torch.tensor(obs, device=device, dtype=torch.float32))
                action_distributions = Categorical(logits=logits)
                acts = action_distributions.sample()
                next_obs, rews, terminateds, truncateds, infos = envs.step(acts.cpu().numpy())

                for i in range(num_envs):
                    if not done[i]:
                        observations[i].append(obs[i])
                        next_observations[i].append(next_obs[i])
                        actions[i].append(acts[i].item())
                        rewards[i].append(rews[i])
                        if terminateds[i] or truncateds[i]:
                            done[i] = True
                            # rewards_to_go = compute_rewards_to_go(rewards[i], gamma=gamma)
                            # discounted_sum_of_rewards[i] = rewards_to_go
                obs = next_obs

        # Flatten nested lists (each env has variable-length episode)
        observations_flat = np.concatenate(observations, axis=0)
        next_observations_flat = np.concatenate(next_observations, axis=0)
        actions_flat = np.concatenate(actions, axis=0)
        rewards_flat = np.concatenate(rewards, axis=0)
        
        # Convert to tensors and move to GPU
        observations = torch.tensor(observations_flat, device=device, dtype=torch.float32)
        next_observations = torch.tensor(next_observations_flat, device=device, dtype=torch.float32)
        actions = torch.tensor(actions_flat, device=device, dtype=torch.int64)
        rewards = torch.tensor(rewards_flat, device=device, dtype=torch.float32)

        # discounted_sum_of_rewards = torch.cat(discounted_sum_of_rewards, dim=0).to(device)
        
        # Train value function
        value.train()
        target = rewards + gamma * value(next_observations).squeeze()
        value_loss = F.huber_loss(value(observations).squeeze(), target)
        value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), max_norm=5.0)
        value_optimizer.step()

        # Calculate advantages
        # advantages = (advantages - advantages.mean())/(advantages.std(unbiased=False) + 1e-8)
        value.eval()
        with torch.no_grad():
            advantages = rewards + gamma * value(next_observations).squeeze() - value(observations).squeeze()
            # advantages = discounted_sum_of_rewards - value(observations).squeeze()
        
        logits = policy(observations)
        act_dist = Categorical(logits=logits)
        log_prob = act_dist.log_prob(actions)
        entropy = act_dist.entropy()
            
        # Compute loss taking the mean over all steps instead of all episodes
        entropy_coef = entropy_coef_start + (entropy_coef_end - entropy_coef_start) * (epoch / num_epochs)
        loss = -(log_prob * advantages).mean() - entropy_coef * entropy.mean()

        # Update policy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        optimizer.step()

        # Test policy after 100 episodes
        if (epoch+1) % test_every == 0:
            env = gym.make("LunarLander-v3")
            test_rewards = []   
            policy.eval() 
            with torch.no_grad():      
                for i in range(num_test_episodes):
                    if test_seed:
                        obs, info = env.reset(seed=test_seed + i)
                    else:
                        obs, info = env.reset(seed=None)
                    done = False
                    episode_reward = 0
                    while not done:
                        logits = policy(torch.tensor(obs, device=device, dtype=torch.float32))
                        action = torch.argmax(logits).item()
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        done = terminated or truncated
                    test_rewards.append(episode_reward)
            print(f"Epoch {epoch + 1} - Average reward: {np.mean(test_rewards)}")

            # Save policy with best rewards
            if np.mean(test_rewards) > best_reward:
                best_reward = np.mean(test_rewards)
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(policy.state_dict(), "checkpoints/best_policy_simple.pt")

def render(policy, num_times):
    env = gym.make("LunarLander-v3", render_mode="human")
    for i in range(num_times):    
        obs, info = env.reset()
        done = False
        tot_reward = 0
        policy.eval()
        with torch.no_grad():
            while not done:
                logits = policy(torch.tensor(obs, dtype=torch.float32))
                action = torch.argmax(logits).item()
                obs, reward, terminated, truncated, info = env.step(action)
                tot_reward += reward
                done = terminated or truncated
            print(f"Total reward for episode {i+1}: {tot_reward}")

if __name__ == "__main__":
    tic = time.time()
    train()
    toc = time.time()
    print(f"Training time: {(toc - tic)/60:.2f} minutes")
    Load best policy
    if os.path.exists("checkpoints/best_policy_simple.pt"):
        policy = Policy()
        policy.load_state_dict(torch.load("checkpoints/best_policy_simple.pt"))
        render(policy, 3)