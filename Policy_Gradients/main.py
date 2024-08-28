import gym, torch, numpy as np, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from policy import Policy
import utils, time
from torch.optim.lr_scheduler import StepLR, ExponentialLR

def train(policy_filename, policy=None):
    # Hyperparameters
    discount_rate = 0.99
    action_sampling = "probabilistic"
    num_trajectories = 512
    num_epochs = 200

    # Create the environment
    env = gym.make('LunarLander-v2', new_step_api=True)
    
    # Initialize the policy if not passed as a parameter
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    if policy is None:
        policy = Policy(state_dim, action_dim)

    optimizer = torch.optim.Adam(params=policy.parameters())
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    losses = []

    for i in range(num_epochs):
        t0 = time.time()
        
        # Sample trajectories from current policy
        observations, actions, rewards, sum_discounted_rewards = utils.get_samples_parallel(
            env, policy, num_samples=num_trajectories, discount_rate=discount_rate, action_sampling=action_sampling
        )
        print(observations.shape)
        t1 = time.time()

        # Train the policy on the samples obtained
        policy, loss = utils.train_step(policy, optimizer, observations, actions, sum_discounted_rewards, num_samples=num_trajectories)
        t2 = time.time()

        # Bookkeeping
        print(f"{i}: Loss {loss:0.4f}, sampling time {t1 - t0:0.4f}, training time {t2 - t1:0.4f}")
        losses.append(loss)
        # scheduler.step()
        # save checkpoint after every 10 epochs
        if i % 10 == 0 and i != 0:
            torch.save(policy.state_dict(), f"{policy_filename}_{i}.pt")

    torch.save(policy.state_dict(), f"{policy_filename}.pt")
    return losses
    
def run(policy_filename, steps):
    # load the policy from saved policy.pt file
    policy = Policy(state_dim=8, action_dim=4)
    policy.load_state_dict(torch.load(f"{policy_filename}.pt"))
    policy.eval()
    env = gym.make('LunarLander-v2', new_step_api=True, render_mode="human")
    observation = env.reset()
    count = 0
    sum_reward = 0
    for i in range(steps):
        with torch.no_grad():
            observation, reward, terminated, truncated, info = env.step(policy(torch.tensor(observation)).argmax().item())
            sum_reward += reward
        
        if terminated or truncated:
            observation = env.reset()
            count += 1
            print(f"{count}: Sum reward {sum_reward}")
            sum_reward = 0
            print(f"{count}: ========== Environment reset ==========")

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
    policy_filename = "checkpoints/policy3"
    policy = None
    losses = train(policy_filename, policy)
    print(losses)
    # run(policy_filename, 3000)
    # observations, actions, rewards, sum_discounted_rewards = get_samples_parallel()
    # print(observations.shape, actions.shape, rewards.shape, sum_discounted_rewards.shape)
    
