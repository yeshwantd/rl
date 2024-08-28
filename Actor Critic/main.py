import gym, torch, numpy as np, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from policy import Policy
from value import Value
import utils

def train(policy_filename, value_filename, policy, value, checkpoint_frequency):
    # Hyperparameters
    discount_rate = 0.99
    action_sampling = "probabilistic"
    num_episodes = 512
    num_epochs = 200

    # Create the environment
    env = gym.make('LunarLander-v2', new_step_api=True)
    
    # Initialize the policy if not passed as a parameter
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    if policy is None:
        policy = Policy(state_dim, action_dim)
    if value is None:
        value = Value(state_dim)

    optimizer_policy = torch.optim.Adam(params=policy.parameters())
    optimizer_value = torch.optim.Adam(params=value.parameters())
    
    policy_losses, value_losses = utils.train(policy, value, optimizer_policy, optimizer_value, env, num_epochs, 
                                              num_episodes, discount_rate, action_sampling, policy_filename, value_filename, checkpoint_frequency)
    return policy_losses, value_losses

def display(policy_filename, episodes):
    # load the policy from saved policy.pt file
    policy = Policy(state_dim=8, action_dim=4)
    policy.load_state_dict(torch.load(f"{policy_filename}.pt"))
    policy.eval()
    env = gym.make('LunarLander-v2', new_step_api=True, render_mode="human")
    observation = env.reset()
    sum_reward = 0
    while episodes > 0:
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float32).view(1,-1)
            observation, reward, terminated, truncated, _ = env.step(policy.get_action(observation))
            sum_reward += reward
        
        if terminated or truncated:
            observation = env.reset()
            episodes -= 1
            print(f"{episodes}: Sum reward {sum_reward}")
            sum_reward = 0

if __name__ == '__main__':
    checkpoint_path = "/Users/ydatta/my_workspace/RL/checkpoints/Actor Critic/"
    policy_filename = checkpoint_path + "policy_ac"
    value_filename = checkpoint_path + "value_ac"
    checkpoint_frequency = 10
    policy = None
    value = None
    train_run = False
    if train_run:
        policy_losses, value_losses = train(policy_filename, value_filename, policy, value, checkpoint_frequency)
        print(policy_losses, value_losses)
    else:
        policy_filename = checkpoint_path + "policy_ac_30"
        display(policy_filename, 3)

    
