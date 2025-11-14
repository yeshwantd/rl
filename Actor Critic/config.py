# Training configuration
SHOW_PLOTS = False
num_epochs = 4000
num_critic_warm_start_epochs = 10
num_episodes = 128
gamma = 0.99
seed = 42
max_episode_steps = 500
num_envs = 8  # parallel gym envs

# Learning rates (lower for more stable training)
critic_lr = 3e-4
actor_lr = 3e-4
# Regularization
entropy_coef_start = 0.1  # Initial entropy bonus coefficient for exploration
entropy_coef_end = 0.01  # Final entropy bonus coefficient
entropy_coef_decay_epochs = 1000  # Number of epochs to decay entropy coefficient
grad_clip = 1.0  # Gradient clipping threshold
