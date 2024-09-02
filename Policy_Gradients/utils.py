import torch, multiprocessing
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

def run_episode(env, policy, num_episodes, verbose=False):
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
        if verbose:
            print("Episode reward: ", episode_reward)
    return episode_rewards

def get_discounted_sum_of_rewards(rewards, discount):
    n = len(rewards)
    sum_of_discounted_rewards = []
    discounts = discount ** torch.arange(n)
    for i in range(n):
        sum_of_discounted_rewards.append(torch.sum(rewards[i:] * discounts[:n-i]))
    return torch.stack(sum_of_discounted_rewards)

def get_sum_of_rewards(rewards):
    out = torch.flip(rewards, dims=(0,))
    out = torch.cumsum(out, dim=0)
    out = torch.flip(out, dims=(0,))
    return out

def run_training_episode(env, policy, discount):
    rewards, sum_discounted_rewards, log_probs = [], [], []
    terminated, truncated = False, False
    observation, _ = env.reset()
    while not (terminated or truncated): # and count < time_steps:
        observation = torch.tensor(observation, dtype=torch.float32).view(1,-1)

        # Get the action and the log probs
        logits = policy(observation)         
        action_probs = F.softmax(logits, dim=1)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        # Perform the action
        next_observation, reward, terminated, truncated, _ = env.step(action.item())

        # Store the results
        rewards.append(reward)
        log_probs.append(log_prob)
        
        # Update the observation
        observation = next_observation
        
    # Create tensors
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
    log_probs = torch.stack(log_probs)
    
    # Get discounted rewards
    sum_discounted_rewards = get_discounted_sum_of_rewards(rewards, discount)

    # Normalize sum of rewards
    sum_discounted_rewards = (sum_discounted_rewards - sum_discounted_rewards.mean())/sum_discounted_rewards.std()

    # Compute the loss
    policy_loss = -(log_probs * sum_discounted_rewards).sum()
    policy_loss.backward()


    return policy_loss

def run_training_episodes(env, policy, discount_rate, num_episodes):
    policy_losses = []
    # if num_processes is None:
    #     num_processes = multiprocessing.cpu_count()
    num_processes = min(multiprocessing.cpu_count(), num_episodes)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            run_training_episode,
            [(env, policy, discount_rate) for _ in range(num_episodes)]
        )
    for result in results:
        policy_losses.append(result)
    policy_losses = torch.stack(policy_losses)
    return policy_losses


def train(env, policy, policy_optimizer, epochs, discount, checkpoint_filepath, checkpoint_frequency, verbose=False):
    losses = []
    for i in range(epochs):
        # Get a sample
        # loss = run_training_episode(env, policy, discount)

        # Get samples
        losses = run_training_episodes(env, policy, discount, 5)
        loss = torch.mean(losses)

        # Backprop
        policy_optimizer.zero_grad()
        loss.backward()
        policy_optimizer.step()
        losses.append(loss.item())

        # Save checkpoint
        if (i+1) % checkpoint_frequency == 0:
            torch.save(policy.state_dict(), f"{checkpoint_filepath}_{i+1}.pt")
            validation_episode_rewards = run_episode(env, policy, num_episodes=10)
            if verbose:
                print(f"Epoch {i+1}: Average Episode Rewards: {np.mean(validation_episode_rewards)}, Episode Rewards Std: {np.std(validation_episode_rewards)}")

    return losses    