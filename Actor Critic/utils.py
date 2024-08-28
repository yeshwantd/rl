import torch, multiprocessing
import torch.nn.functional as F
import numpy as np
import time
# from memory_profiler import profile

from sklearn.preprocessing import normalize


def get_trajectory(env, policy, discount, action_sampling):
    observations, actions, rewards, sum_discounted_rewards = [], [], [], []
    terminated, truncated = False, False
    observation = env.reset()
    while not (terminated or truncated): 
        observation = torch.tensor(observation, dtype=torch.float32).view(1,-1)
        
        if torch.cuda.is_available():
            device = "cuda"
            policy.to(device)
            observation = observation.to(device)
        else:
            device = "cpu"
        
        # Get the action
        policy.eval()
        with torch.no_grad():
            action = policy.get_action(observation, action_sampling)
        next_observation, reward, terminated, truncated, _ = env.step(action)

        # Store the results
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        
        observation = next_observation
        if sum(rewards) < -250:
            break

    observations = torch.vstack(observations)
    actions = torch.tensor(np.array(actions), dtype=torch.int64)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
    
    sum_rewards = get_sum_rewards(rewards)

    # Get discounted rewards and normalize it to reduce variance
    # sum_discounted_rewards = get_discounted_rewards(rewards, discount)
    # sum_discounted_rewards = normalize(np.array(sum_discounted_rewards).reshape(1,-1)).reshape(-1)

    return observations, actions, rewards, sum_rewards

def get_samples(env, policy, num_episodes, discount_rate, action_sampling):
    observations, actions, rewards, sum_discounted_rewards = [], [], [], []
    for i in range(num_episodes):
        o, a, r, dr = get_trajectory(env, policy, discount_rate, action_sampling)        
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

def get_discounted_rewards(rewards, discount):
    n = len(rewards)
    sum_of_discounted_rewards = []
    rewards = np.array(rewards)
    discounts = discount ** np.arange(n)
    for i in range(n):
        sum_of_discounted_rewards.append(np.sum(rewards[i:] * discounts[:n-i]))
    return sum_of_discounted_rewards

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

def train(policy, value, optimizer_policy, optimizer_value, env, num_epochs, num_trajectories, discount, action_sampling, 
          policy_filepath, value_filepath, chekpoint_frequency):
    policy_losses, value_losses = [], []

    for i in range(num_epochs):
        t0 = time.time()

        # Sample trajectories from current policy  
        observations, actions, rewards, sum_discounted_rewards = get_samples_parallel(env, policy, num_trajectories, discount, action_sampling)
        t1 = time.time()
        # Train the Value function
        value, value_loss = train_value_step(value, optimizer_value, observations, sum_discounted_rewards)

        # Train the policy on the samples obtained
        policy, policy_loss = train_policy_step(policy, value, optimizer_policy, observations, actions, rewards, gamma=0.99, num_samples=num_trajectories)
        t2 = time.time()

        print(f"{i}: Policy Loss {policy_loss:0.4f}, Value Loss {value_loss:0.4f}, sampling time {t1 - t0:0.4f}, training time {t2 - t1:0.4f}")
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)

        if i % chekpoint_frequency == 0 and i != 0:
            torch.save(policy.state_dict(), f"{policy_filepath}_{i}.pt")
            torch.save(value.state_dict(), f"{value_filepath}_{i}.pt")

    torch.save(policy.state_dict(), f"{policy_filepath}.pt")
    torch.save(value.state_dict(), f"{value_filepath}.pt")
    return policy_losses, value_losses
