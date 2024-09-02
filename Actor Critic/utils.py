import torch, multiprocessing
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import time
# from memory_profiler import profile

from sklearn.preprocessing import normalize

def run_episode(env, policy, num_episodes):
    episode_rewards = []
    for _ in range(num_episodes):
        observation, _ = env.reset()
        terminated, truncated = False, False
        episode_reward = 0
        while not (terminated or truncated):
            observation = torch.tensor(observation).view(1,-1)
            with torch.no_grad():
                action = policy.action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            observation = next_observation
        episode_rewards.append(episode_reward)
    return episode_rewards

def get_discounted_rewards(rewards, discount):
    n = len(rewards)
    sum_of_discounted_rewards = []
    discounts = discount ** torch.arange(n)
    for i in range(n):
        sum_of_discounted_rewards.append(torch.sum(rewards[i:] * discounts[:n-i]))
    return torch.tensor(np.array(sum_of_discounted_rewards))

def run_training_episode(env, policy, value, discount):
    observations, rewards, sum_discounted_rewards, log_probs = [], [], [], []
    
    terminated, truncated = False, False
    observation, _ = env.reset()
    while not (terminated or truncated): 
        observation = torch.tensor(observation).view(1,-1)

        # Get the action and the log probs
        logits = policy(observation)
        action_probs = F.softmax(logits, dim=1)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        # Perform the action
        next_observation, reward, terminated, truncated, _ = env.step(action.item())

        # Store the results
        observations.append(observation)
        rewards.append(reward)
        log_probs.append(log_prob)
        
        observation = next_observation
        
    observations = torch.vstack(observations)
    rewards = torch.tensor(np.array(rewards))
    log_probs = torch.cat(log_probs)
    
    
    # Get discounted rewards 
    # sum_rewards = get_sum_rewards(rewards)
    sum_discounted_rewards = get_discounted_rewards(rewards, discount)

    # Normalize the rewards to reduce variance
    # sum_discounted_rewards = F.normalize(sum_discounted_rewards, dim=0)
    sum_discounted_rewards = (sum_discounted_rewards - sum_discounted_rewards.mean())/sum_discounted_rewards.std()
    
    # Get the value of the states
    values = value(observations)
    values = values.squeeze()

    # Get the value and policy losses
    advantage = sum_discounted_rewards - values.detach()
    value_loss = F.smooth_l1_loss(values, sum_discounted_rewards)
    policy_loss = -(log_probs * advantage).sum()
    
    # return observations, actions, rewards, sum_discounted_rewards, values
    return value_loss, policy_loss
    
def get_samples(env, policy, num_episodes, discount_rate, action_sampling):
    observations, actions, rewards, sum_discounted_rewards = [], [], [], []
    for i in range(num_episodes):
        o, a, r, dr = run_training_episode(env, policy, discount_rate, action_sampling)        
        observations.append(o)
        actions.append(a)
        rewards.append(r)
        sum_discounted_rewards.append(dr)         
    return torch.vstack(observations), torch.cat(actions), torch.cat(rewards), torch.cat(sum_discounted_rewards)

def get_samples_parallel(env, policy, num_episodes, discount_rate, action_sampling, num_processes=None):
    observations, actions, rewards, sum_discounted_rewards = [], [], [], []
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    num_processes = min(num_processes, num_episodes)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            get_samples,
            [(env, policy, num_episodes//num_processes, discount_rate, action_sampling) for _ in range(num_episodes)]
        )
    for result in results:
        o, a, r, dr = result 
        observations.append(o)
        actions.append(a)
        rewards.append(r)
        sum_discounted_rewards.append(dr)
    observations = torch.vstack(observations)
    actions = torch.cat(actions)
    rewards = torch.cat(rewards)
    sum_discounted_rewards = torch.cat(sum_discounted_rewards)

    return observations, actions, rewards, sum_discounted_rewards

def get_sum_rewards(rewards):
    out = torch.flip(rewards, dims=(0,))
    out = torch.cumsum(out, dim=0)
    out = torch.flip(out, dims=(0,))
    return out

def train_policy_step(policy, value, optimizer, observations, actions, rewards, gamma, num_samples):
    if torch.cuda.is_available():
        device = "cuda"
        policy.to(device)
        value.to(device)
        observations = observations.to(device)
        rewards = rewards.to(device)
        actions = actions.to(device)
    else:
        device = "cpu"
    policy.train()
    logits = policy(observations)
    value.eval()
    with torch.no_grad():
        V_curr = value(observations).squeeze()
    V_next = torch.cat([V_curr[1:], V_curr[-1:]])
    advantage = rewards + gamma*V_next - V_curr
    loss = (1/num_samples) * (F.cross_entropy(logits, actions, reduction="none")*advantage).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return policy, loss.item()

def train_value_step(value, optimizer, observations, sum_discounted_rewards): 
    if torch.cuda.is_available():
        device = "cuda"
        value.to(device)
        observations = observations.to(device)
        sum_discounted_rewards = sum_discounted_rewards.to(device)
    else:
        device = "cpu"
    value.train()
    loss = F.mse_loss(value(observations), sum_discounted_rewards.view(-1,1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return value, loss.item()

def train(policy, value, policy_optimizer, value_optimizer, env, num_epochs, discount, policy_filepath, value_filepath, checkpoint_frequency):
    policy_losses, value_losses = [], []

    for i in range(num_epochs):

        # Sample a single trajectory
        # observations, actions, rewards, sum_discounted_rewards = get_samples_parallel(env, policy, num_trajectories, discount, action_sampling)
        value_loss, policy_loss = run_training_episode(env, policy, value, discount)

        # Train the Value function
        # value, value_loss = train_value_step(value, optimizer_value, observations, sum_discounted_rewards)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # Train the Policy
        # policy, policy_loss = train_policy_step(policy, value, policy_optimizer, observations, actions, rewards, gamma=0.99, num_samples=num_trajectories)
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())

        if (i+1) % checkpoint_frequency == 0:
            torch.save(policy.state_dict(), f"{policy_filepath}_{i+1}.pt")
            torch.save(value.state_dict(), f"{value_filepath}_{i+1}.pt")
            policy.eval()
            validation_rewards = run_episode(env, policy, 10)
            print(f"{i+1}: Average Rewards: {np.mean(validation_rewards)}, Episode Rewards: {validation_rewards}")

    return policy_losses, value_losses
