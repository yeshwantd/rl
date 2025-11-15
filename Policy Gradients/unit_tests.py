"""
Unit tests for Policy Gradient implementation

Run with: python -m pytest unit_tests.py -v
Or: python unit_tests.py
"""
import unittest
import torch
import numpy as np
import sys
import os
import gymnasium as gym

# Add the current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from models import BasicPolicy
from utils import compute_rewards_to_go, test_policy, visualize_policy
from policy_gradient import collect_data

class TestUtils(unittest.TestCase):
    """Test utility functions"""

    def test_compute_rewards_to_go(self):
        """Test rewards to go computation"""
        rewards = [1, 2, 3]
        gamma = 0.9
        expected = np.array([1 + 2 * 0.9 + 3 * 0.9 * 0.9, 2 + 3 * 0.9, 3])
        result = compute_rewards_to_go(rewards, gamma)
        self.assertTrue(np.allclose(result, expected))

    def test_test_policy(self):
        """Test test_policy function"""
        env = gym.make("LunarLander-v3")
        policy = BasicPolicy()
        result = test_policy(env, policy)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) == 10)
        self.assertTrue(all(isinstance(x, float) for x in result))

class TestBasicPolicy(unittest.TestCase):
    """Test BasicPolicy class"""

    def test_forward(self):
        """Test forward pass"""
        policy = BasicPolicy()
        x = torch.randn(8)
        output = policy.forward(x)
        self.assertEqual(output.shape, (4,))

    def test_forward_batch(self):
        """Test forward pass with batch input"""
        policy = BasicPolicy()
        batch_size = 10
        x = torch.randn(batch_size, 8)
        output = policy.forward(x)
        self.assertEqual(output.shape, (batch_size, 4))

    def test_get_action_distribution(self):
        """Test get_action_distribution method"""
        policy = BasicPolicy()
        x = torch.randn(8)
        action_distribution = policy.get_action_distribution(x)
        self.assertEqual(action_distribution.shape, (4, ))
        self.assertTrue(torch.all(action_distribution >= 0))
        self.assertTrue(torch.all(action_distribution <= 1))
        self.assertTrue(torch.allclose(action_distribution.sum(), torch.tensor(1.0)))
    
    def test_get_action(self):
        """Test get_action method"""
        policy = BasicPolicy()
        x = torch.randn(8)
        action = policy.get_action(x)
        self.assertTrue(action in [0, 1, 2, 3])
        self.assertTrue(isinstance(action, int))

class TestData(unittest.TestCase):
    """Test data generation"""
    def test_collect_data(self):
        """Test data collection"""
        policy = BasicPolicy()
        num_envs = 4
        seed = 47
        max_episode_steps = 10
        log_probs, rewards, entropies = collect_data(policy, num_envs, max_episode_steps, seed)
        self.assertEqual(len(rewards), num_envs)
        self.assertEqual(len(log_probs), num_envs)
        self.assertEqual(len(entropies), num_envs)
        self.assertEqual(len(rewards[0]), max_episode_steps)


def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()