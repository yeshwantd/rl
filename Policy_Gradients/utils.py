import torch, multiprocessing
import torch.nn.functional as F
import numpy as np

from sklearn.preprocessing import normalize


def get_trajectory(env, policy, discount=0.9, sampling="deterministic"):
    observations, actions, rewards, sum_discounted_rewards = [], [], [], [] # State, Action, Reward, Sum of discounted rewards
    terminated, truncated = False, False
    observation = env.reset()
    while (not (terminated or truncated)): # and count < time_steps:
        observation = torch.tensor(observation, dtype=torch.float32)
        observations.append(observation)

        # Get the action from the logits
        policy.eval()
        with torch.no_grad():
            logits = policy(observation.view(1,-1))         
        if sampling == "deterministic":
            action = torch.argmax(logits).item()
        else:
            action = torch.multinomial(F.softmax(logits, dim=1), num_samples=1).item()

        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Store the results
        actions.append(action)
        rewards.append(reward)
        if sum(rewards) < -250:
            break

    # Get discounted rewards and normalize it to reduce variance
    sum_discounted_rewards = get_discounted_rewards(rewards, discount)
    sum_discounted_rewards = normalize(np.array(sum_discounted_rewards).reshape(1,-1)).reshape(-1)

    return torch.vstack(observations), torch.tensor(actions), torch.tensor(rewards), torch.tensor(sum_discounted_rewards)

def get_samples(env, policy, num_samples, discount=0.9):
    observations, actions, rewards, sum_discounted_rewards = [], [], [], []
    for i in range(num_samples):
        o, a, r, dr = get_trajectory(env, policy, discount)        
        observations.append(o)
        actions.append(a)
        rewards.append(r)
        sum_discounted_rewards.append(dr)         
    return torch.vstack(observations), torch.cat(actions), torch.cat(rewards), torch.cat(sum_discounted_rewards)

def get_samples_parallel(env, policy, num_samples, discount=0.9, num_processes=None):
    observations, actions, rewards, sum_discounted_rewards = [], [], [], []
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    num_processes = min(num_processes, num_samples)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            get_samples,
            [(env, policy, num_samples//num_processes, discount) for _ in range(num_samples)]
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

def get_discounted_rewards(rewards, discount=0.9):
    n = len(rewards)
    sum_of_discounted_rewards = []
    rewards = np.array(rewards)
    discounts = discount ** np.arange(n)
    for i in range(n):
        sum_of_discounted_rewards.append(np.sum(rewards[i:] * discounts[:n-i]))
    return sum_of_discounted_rewards

def train_step(policy, optimizer, observations, actions, sum_discounted_rewards, num_samples):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    policy.to(device)
    policy.train()
    observations = observations.to(device)
    logits = policy(observations)
    sum_discounted_rewards = sum_discounted_rewards.to(device)
    actions = actions.to(device)
    loss = (1/num_samples) * (F.cross_entropy(logits, actions, reduction="none")*sum_discounted_rewards).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return policy, loss.item()
        