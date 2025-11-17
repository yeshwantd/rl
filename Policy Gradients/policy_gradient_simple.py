import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    T = len(rewards) 
    rewards = torch.tensor(rewards, dtype=torch.float32)
    discounts = gamma ** torch.arange(T)
    x = rewards * discounts
    y = torch.flip(torch.cumsum(torch.flip(x, dims=[0]), dim=0), dims=[0])
    return y/discounts


# Train
def train():
    # Variables
    num_epochs = 1000
    num_episodes_per_epoch = 64
    test_every = 100 # tests every n epochs
    seed = 42
    test_seed = 55
    num_test_episodes = 10
    best_reward = 0

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize
    policy = Policy()
    env = gym.make("LunarLander-v3")
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    # Train
    for epoch in range(num_epochs):
        policy.train()
        observations, actions, advantages = [], [], []
        
        # Collect data - disable autograd
        with torch.no_grad():
            for episode in range(num_episodes_per_epoch):
                rewards = []
                obs, info = env.reset(seed=seed + epoch * num_episodes_per_epoch + episode)
                done = False
            
                while not done:
                    logits = policy(torch.tensor(obs, dtype=torch.float32))
                    action = Categorical(logits=logits).sample().item()
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    observations.append(obs)
                    actions.append(action)
                    rewards.append(reward)
                    obs = next_obs
                    done = terminated or truncated

                # Compute rewards to go per episode
                rewards_to_go = compute_rewards_to_go(rewards, gamma=0.99)
                advantages.append(rewards_to_go)

        # Convert to tensors and move to GPU
        policy = policy.to(device)
        observations = torch.tensor(np.array(observations), device=device, dtype=torch.float32)
        actions = torch.tensor(np.array(actions), device=device, dtype=torch.int64)
        logits = policy(observations)
        act_dist = Categorical(logits=logits)
        log_prob = act_dist.log_prob(actions)
        entropy = act_dist.entropy()
        
        # Subtract baseline 
        advantages = torch.cat(advantages, dim=0).to(device)
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
        
        # Mini batch update
        batch_size = 512
        n_steps = log_prob.size(0)
        idxs = torch.randperm(n_steps)

        for start in range(0, n_steps, batch_size):
            end = start + batch_size
            batch = idxs[start:end]

            # Compute loss taking the mean over all steps instead of all episodes
            loss = -(log_prob[batch] * advantages[batch]).mean() - 0.01 * entropy[batch].mean()

            # Update policy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
            optimizer.step()

        # Test policy after 100 episodes
        if (epoch+1) % test_every == 0:
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
                torch.save(policy.state_dict(), "checkpoints/best_policy_simple.pt")

def render(policy):
    env = gym.make("LunarLander-v3", render_mode="human")
    obs, info = env.reset()
    done = False
    while not done:
        logits = policy(torch.tensor(obs, dtype=torch.float32))
        action = torch.argmax(logits).item()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

if __name__ == "__main__":
    train()
    # Load best policy
    if os.path.exists("checkpoints/best_policy_simple.pt"):
        policy = Policy()
        policy.load_state_dict(torch.load("checkpoints/best_policy_simple.pt"))
        render(policy)