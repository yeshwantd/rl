import gymnasium as gym
from policy import Policy
from value import Value
import utils
import torch

def get_action():
    env = gym.envs.make('LunarLander-v2')
    observation, _ = env.reset()
    policy = Policy(state_dim=8, action_dim=4)
    return policy.action(torch.tensor(observation).view(1,-1))

def run_episode():
    env = gym.envs.make('LunarLander-v2')
    policy = Policy(state_dim=8, action_dim=4)
    value = Value(state_dim=8)
    return utils.run_training_episode(env, policy, value, discount=0.99)

def get_samples():
    env = gym.make('LunarLander-v2', new_step_api=True)
    policy = Policy(state_dim=8, action_dim=4)
    return utils.get_samples(env, policy, num_samples=5)

def get_samples_parallel():
    # envs = [gym.make('LunarLander-v2', new_step_api=True) for i in range(10)] 
    env = gym.make('LunarLander-v2', new_step_api=True)
    policy = Policy(state_dim=8, action_dim=4)
    return utils.get_samples_parallel(env, policy, num_samples=20)

def get_trajectory():
    env = gym.make('LunarLander-v2', new_step_api=True)
    policy = Policy(state_dim=8, action_dim=4)
    # policy.load_state_dict(torch.load("policy.pt"))
    return utils.get_trajectory(env, policy, time_steps=12, discount=0.9)

if __name__ == '__main__':
    # print(get_action())
    value_loss, policy_loss = run_episode()
    print(value_loss, policy_loss)