import random
import numpy as np

import torch

# Code structure based on lab4 from Reinforcement Learning course at University of Amsterdam

def get_epsilon(it,  min_eps, max_eps, annealing_time):
    progress = it/annealing_time
    epsilon = max(max_eps - (max_eps - min_eps) * progress, min_eps)
    
    return epsilon


class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, num_actions):
        self.Q = Q
        self.epsilon = None  # need to set epsilon to avoid confusion
        self.num_actions = num_actions
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        # This piece of code was copied and altered from our version of lab4 - Yke
        with torch.no_grad():
            obs_tensor = torch.tensor(obs).type(torch.FloatTensor)
            model_output = self.Q(obs_tensor)
            a_greedy = torch.argmax(model_output).item()
        probs = [self.epsilon / self.num_actions for i in range(self.num_actions)]

        probs[a_greedy] += 1 - self.epsilon
        assert np.allclose(sum(probs), 1), "Probabilies should sum to 1"
        return np.random.choice(range(self.num_actions), p=probs)
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon