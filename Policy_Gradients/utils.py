import torch, multiprocessing
import torch.nn.functional as F
import numpy as np

from sklearn.preprocessing import normalize


def get_trajectory(env, policy, time_steps, discount=0.9):
    observations, actions, rewards, sum_discounted_rewards = [], [], [], [] # State, Action, Reward, Sum of discounted rewards
    terminated, truncated, count = False, False, 0
    observation = env.reset()
    while (not (terminated or truncated)) and count < time_steps:
        observations.append(observation)
        policy.eval()
        with torch.no_grad():
            logits = policy(torch.tensor(observation, dtype=torch.float32).view(1,-1)) 
        
        # Sample an action probabilitistically
        # action = torch.multinomial(
        #     F.softmax(logits, dim=1), num_samples=1
        # ).item()

        # Sample an action deterministically
        action = torch.argmax(logits).item()

        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Store the results
        actions.append(action)
        rewards.append(reward)
        if reward >= 100:
            print(f"Reward {reward} is >= 100")
        count += 1
        if sum(rewards) < -250:
            break
    if sum(rewards) > 100:
        print(f"Sum of reward {sum(rewards)} is larger than 100")

    # Get discounted rewards and normalize it to reduce variance
    sum_discounted_rewards = get_discounted_rewards(rewards, discount)
    sum_discounted_rewards = normalize(np.array(sum_discounted_rewards).reshape(1,-1)).reshape(-1)

    # Pad the trajectory so we have the same number of observations and actions for training
    observations, actions, sum_discounted_rewards = pad_trajectory(observations,actions, sum_discounted_rewards, time_steps)
    return observations, actions, rewards, sum_discounted_rewards

def pad_trajectory(observations, actions, sum_discounted_rewards, time_steps, neutral_action=0):
    # Pad the trajectory for use in training
    observations = np.array(observations)
    observations = np.vstack([observations, np.tile(observations[-1], (time_steps - observations.shape[0], 1))])
    actions = np.pad(np.array(actions), (0, time_steps - len(actions)), "constant", constant_values=neutral_action)
    sum_discounted_rewards = np.pad(np.array(sum_discounted_rewards), (0, time_steps - len(sum_discounted_rewards)), "constant", constant_values=0.0)
    return observations, actions, sum_discounted_rewards

def get_samples(env, policy, num_samples, time_steps, discount=0.9):
    observations, actions, rewards, sum_discounted_rewards = [], [], [], []
    for i in range(num_samples):
        o, a, r, dr = get_trajectory(env, policy, time_steps, discount)        
        observations.append(o)
        actions.append(a)
        rewards.append(r)
        sum_discounted_rewards.append(dr)         
    return observations, actions, rewards, sum_discounted_rewards

def get_samples_parallel(envs, policy, num_samples, time_steps, discount=0.9, num_processes=None):
    observations, actions, rewards, sum_discounted_rewards = [], [], [], []
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            get_samples,
            [(env, policy, num_samples//num_processes, time_steps, discount) for env in envs]
        )
    for result in results:
        o, a, r, dr = result 
        observations.append(o)
        actions.append(a)
        rewards.append(r)
        sum_discounted_rewards.append(dr)
    observations = np.vstack(observations)
    actions = np.vstack(actions)
    sum_discounted_rewards = np.vstack(sum_discounted_rewards)

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
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    policy.to(device)
    policy.train()
    # observations has dimensions num_trajectories X time_steps X state_dim (which is 8 for lunar lander)
    N, T, _ = observations.shape
    observations = torch.tensor(observations).view(-1,observations[0].shape[-1])
    observations = observations.to(device)
    logits = policy(observations)
    sum_discounted_rewards = torch.tensor(sum_discounted_rewards).view(-1)
    sum_discounted_rewards = sum_discounted_rewards.to(device)
    actions = torch.tensor(actions).view(-1)
    actions = actions.to(device)
    loss = (1/N) * (F.cross_entropy(logits, actions, reduction="none")*sum_discounted_rewards).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return policy, loss.item()
        