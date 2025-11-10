import numpy as np
import matplotlib.pyplot as plt
import torch



def compute_rewards_to_go(rewards, gamma):
    """
    Compute the rewards-to-go for a given sequence of rewards.

    Args:
        rewards (list): A list of rewards.
        gamma (float): The discount factor.

    Returns:
        list: A list of rewards-to-go for each time step.
    """
    T = len(rewards) 
    gamma_vector = np.array([gamma**t for t in range(T)])
    reversed_gamma_vector = np.flip(gamma_vector)
    rewards_to_go = np.convolve(rewards, reversed_gamma_vector)[-T:]
    return rewards_to_go


def test_policy(env, policy, n_episodes=10, device='cpu'):
    """
    Test the policy on the environment.

    Args:
        env (gym.Env): The environment to test the policy on.
        policy (Policy): The policy to test.
        n_episodes (int, optional): The number of episodes to test the policy on. Defaults to 10.
        device (str): The device to run the policy on.
    """
    episode_rewards = []
    policy.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            done = False
            observation, info = env.reset()
            episode_reward = 0        
            while not done:
                obs_tensor = torch.tensor(observation, dtype=torch.float32).to(device)
                action = policy.get_action(obs_tensor)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            episode_rewards.append(episode_reward)
    return episode_rewards

def visualize_policy(env, policy, num_episodes=5, device='cpu'):
    """
    Visualize the policy on the environment.

    Args:
        env (gym.Env): The environment to visualize the policy on.
        policy (Policy): The policy to visualize.
        device (str): The device to run the policy on.
    """
    policy.eval()
    with torch.no_grad():
        for _ in range(num_episodes):
            done = False
            observation, info = env.reset()
            total_reward = 0
            while not done:
                obs_tensor = torch.tensor(observation, dtype=torch.float32).to(device)
                action = policy.get_action(obs_tensor)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                env.render()
                done = terminated or truncated
            print(f"Episode reward: {total_reward}")
    env.close()


def plot(x, xlabel, ylabel, title=None):
    plt.plot(x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.show()