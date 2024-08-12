import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import normalize


def get_trajectory(env, initial_observation, policy, T=1024, discount=0.9):
    observations, actions, rewards, sum_discounted_rewards = [], [], [], [] # State, Action, Reward, Sum of discounted rewards
    terminated, truncated, count = False, False, 0
    observation = initial_observation
    while (not (terminated or truncated)) and count < T:
        observations.append(observation)
        policy.eval()
        with torch.no_grad():
            logits = policy(torch.tensor(observation, dtype=torch.float32).view(1,-1)) 
        action = torch.multinomial(
            F.softmax(logits, dim=1), num_samples=1
        ).item()
        observation, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        count += 1
    sum_discounted_rewards = get_discounted_rewards(rewards, discount)
    sum_discounted_rewards = normalize(np.array(sum_discounted_rewards).reshape(1,-1)).reshape(-1)
    observations, actions, sum_discounted_rewards = pad_trajectory(observations,actions, sum_discounted_rewards, T)
    return observations, actions, rewards, sum_discounted_rewards

def pad_trajectory(observations, actions, sum_discounted_rewards, T=1024, neutral_action=0):
    # Pad the trajectory for use in training
    observations = np.array(observations)
    observations = np.vstack([observations, np.tile(observations[-1], (T - observations.shape[0], 1))])
    actions = np.pad(np.array(actions), (0, T - len(actions)), "constant", constant_values=neutral_action)
    sum_discounted_rewards = np.pad(np.array(sum_discounted_rewards), (0, T - len(sum_discounted_rewards)), "constant", constant_values=0.0)
    return observations, actions, sum_discounted_rewards

def get_samples(env, policy, num_samples=1024, T=1024, discount=0.9):
    observations, actions, rewards, sum_discounted_rewards = [], [], [], []
    for i in range(num_samples):
        initial_observation = env.reset()
        o, a, r, dr = get_trajectory(env, initial_observation, policy)
        observations.append(o)
        actions.append(a)
        rewards.append(r)
        sum_discounted_rewards.append(dr) 
    return observations, actions, rewards, sum_discounted_rewards

def get_discounted_rewards(rewards, discount=0.9):
    n = len(rewards)
    sum_of_discounted_rewards = []
    rewards = np.array(rewards)
    discounts = discount ** np.arange(n)
    for i in range(n):
        sum_of_discounted_rewards.append(np.sum(rewards[i:] * discounts[:n-i]))
    return sum_of_discounted_rewards

def train_step(policy, optimizer, observations, actions, sum_discounted_rewards):
    policy.train()
    observations = torch.tensor(observations).view(-1,observations[0].shape[-1])
    logits = policy(observations)
    sum_discounted_rewards = torch.tensor(sum_discounted_rewards).view(-1)
    loss = (F.cross_entropy(logits, torch.tensor(actions).view(-1), reduction="none")*sum_discounted_rewards).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return policy, loss.item()
        