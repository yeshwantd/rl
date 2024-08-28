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
