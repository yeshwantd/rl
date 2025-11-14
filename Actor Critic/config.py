# Training configuration
SHOW_PLOTS = False
num_epochs = 4000
num_critic_warm_start_epochs = 10
num_episodes = 512
gamma = 0.99
seed = 42
max_episode_steps = 500
num_envs = 512  # parallel gym envs

# Learning rates (lower for more stable training)
critic_lr = 3e-4
actor_lr = 1e-4
# Regularization
param_reg_coef = 0.01  # Coefficient for parameter regularization in critic training
entropy_coef = 0.01  # Entropy bonus coefficient for exploration
grad_clip = 1.0  # Gradient clipping threshold
