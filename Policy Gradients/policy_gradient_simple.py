import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import matplotlib.pyplot as plt
import random
import os, time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Aysnc vector env
# def make_envs(num_envs, seed):
#     def make_env(env_seed):
#         def thunk():
#             env = gym.make("LunarLander-v3")
#             env.reset(seed=env_seed)
#             return env
#         return thunk
#     envs = AsyncVectorEnv([make_env(seed + i) for i in range(num_envs)])
#     return envs

def make_envs(num_envs):
    def make_env():
        def thunk():
            env = gym.make("LunarLander-v3")
            return env
        return thunk
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
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
    num_epochs = 1000
    num_episodes_per_epoch = 64
    num_envs = 64
    test_every = 100 # tests every n epochs
    seed = None
    test_seed = None
    num_test_episodes = 10
    best_reward = 0

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize
    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    # env = gym.make("LunarLander-v3")
    envs = make_envs(num_envs)
    
    # Train
    for epoch in range(num_epochs):
        policy.train()
        observations = [[] for _ in range(num_envs)]
        actions = [[] for _ in range(num_envs)]
        rewards  = [[] for _ in range(num_envs)]
        advantages = [[] for _ in range(num_envs)]
        
        # Collect data - disable autograd
        with torch.no_grad():
            seeds = [seed + epoch * num_envs + i for i in range(num_envs)]
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
                        actions[i].append(acts[i].item())
                        rewards[i].append(rews[i])
                        if terminateds[i] or truncateds[i]:
                            done[i] = True
                            rewards_to_go = compute_rewards_to_go(rewards[i], gamma=0.99)
                            advantages[i] = rewards_to_go

                obs = next_obs

        # Flatten nested lists (each env has variable-length episode)
        observations_flat = np.concatenate(observations, axis=0)
        actions_flat = np.concatenate(actions, axis=0)
        
        # Convert to tensors and move to GPU
        observations = torch.tensor(observations_flat, device=device, dtype=torch.float32)
        actions = torch.tensor(actions_flat, device=device, dtype=torch.int64)
        advantages = torch.cat(advantages, dim=0).to(device)
        
        advantages = (advantages - advantages.mean())/(advantages.std(unbiased=False) + 1e-8)
        
        logits = policy(observations)
        act_dist = Categorical(logits=logits)
        log_prob = act_dist.log_prob(actions)
        entropy = act_dist.entropy()
            
        # Compute loss taking the mean over all steps instead of all episodes
        loss = -(log_prob * advantages).mean() - 0.01 * entropy.mean()

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
                        obs, info = env.reset(seed=random.randint(0, 1000000))
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
    # Load best policy
    if os.path.exists("checkpoints/best_policy_simple.pt"):
        policy = Policy()
        policy.load_state_dict(torch.load("checkpoints/best_policy_simple.pt"))
        render(policy, 3)