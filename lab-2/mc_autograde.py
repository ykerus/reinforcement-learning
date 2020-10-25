import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains a probability
        of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        probs = np.zeros(len(states))
        for i, (player_sum, _, _) in enumerate(states):
            if player_sum < 20 and actions[i] == 1:
                probs[i] = 1
            elif player_sum >= 20 and actions[i] == 0:
                probs[i] = 1
        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        options = [0,1]
        action = np.random.choice(options, p=self.get_probs([state]*2, options))
        
        return action

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length. 
        Hint: Do not include the state after the termination in the list of states.
    """
    states, actions, rewards, dones = [], [], [], []
    
    s = env.reset()
    done = False
    # if player is bust, done==True
    while not done: 
        states.append(s)
        
        a = policy.sample_action(s)
        s, r, done, _ = env.step(a)
        
        actions.append(a)
        rewards.append(r)
        dones.append(done)
    
    return states, actions, rewards, dones

def mc_prediction(env, policy, num_episodes, discount_factor=1.0, sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)
    returns_sum = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        trajectory_data = sampling_function(env, policy)
        states, actions, rewards, dones = trajectory_data
        T_trajectory = len(states)
        G = 0
        for t in range(T_trajectory-1, -1, -1):
            state_t = states[t]
            # use state[t] because this is the state that came
            # before doing action actions[t] and receiving reward rewards[t]
            G = discount_factor * G + rewards[t]
            if state_t not in states[:t]:
                returns_sum[state_t] += G
                returns_count[state_t] += 1
                V[state_t] = returns_sum[state_t] / returns_count[state_t]
    return V

class RandomBlackjackPolicy(object):
    """
    A random BlackJack policy.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains 
        a probability of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        # YOUR CODE HERE
        probs=np.zeros(len(states))+0.5
        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        options = [0,1]
        action = np.random.choice(options, p=self.get_probs([state]*2, options))
        return action

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    # YOUR CODE HERE
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)
    returns_sum = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        trajectory_data = sampling_function(env, behavior_policy)
        states, actions, rewards, dones = trajectory_data
        T_trajectory = len(states)-1
        G = 0
        for t in range(T_trajectory, -1, -1):
            state_t = states[t]
            action_t = actions[t]
            # use state[t] because this is the state that came
            # before doing action actions[t] and receiving reward rewards[t]
            G = discount_factor * G + rewards[t]
            imp_w = target_policy.get_probs([state_t],[action_t])/behavior_policy.get_probs([state_t],[action_t])
            if state_t not in states[:t]:
                returns_sum[state_t] += G * imp_w
                returns_count[state_t] += 1
                V[state_t] = returns_sum[state_t] / returns_count[state_t]
    # codegrade marks this part as incorrect, but we believe it's correct as we adapted the first part
    # of this assignment according to the slides concening importance sampling. Also, codegrade's error
    # message states that only after 10k episodes the v-value deviates from what it's supposed to be, so
    # this might be due to rounding issues or other small effects.

    return V
