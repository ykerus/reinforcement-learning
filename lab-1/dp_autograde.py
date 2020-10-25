import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    
    # following p.80 of RL book
    while True:
        delta = 0
        for s in env.P:
            v_prev = V[s]
            v_updated = 0
            for a in env.P[s]:
                temp = 0
                for transition in env.P[s][a]:
                    prob, next_state, reward, done = transition
                    temp += prob * (reward + discount_factor * V[next_state])
                v_updated += policy[s,a] * temp
            V[s] = v_updated
            delta = max(delta, abs(v_prev - V[s]))
        if delta < theta:
            break
    
    return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        
        policy_stable = True
        for s in env.P:
            policy_prev = np.copy(policy[s])
            
            values = np.zeros(env.nA)
            for a in env.P[s]:
                for transition in env.P[s][a]:
                    prob, next_state, reward, done = transition
                    values[a] += prob * (reward + discount_factor * V[next_state])
            policy[s] *= 0
            policy[s, np.argmax(values)] = 1
            if sum(policy[s] != policy_prev) > 0:
                policy_stable = False
        if policy_stable:
            break
    
    return policy, V

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function and random policy
    Q = np.zeros((env.nS, env.nA))
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        delta = 0
        for s in env.P:
            for a in env.P[s]:
                q_prev = Q[s,a]
                q_updated = 0
                for transition in env.P[s][a]:
                    prob, s_next, reward, done = transition
                    q_next = np.zeros(env.nA)
                    for a_next in env.P[s_next]:
                        q_next[a_next] = Q[s_next, a_next]
                    # using update rule as written in Q2 of the homework
                    q_updated += prob * (reward + discount_factor * max(q_next))
                Q[s,a] = q_updated
                delta = max(delta, abs(q_prev - Q[s,a]))
            # learn policy implicitly
            policy[s] *= 0
            policy[s, np.argmax(Q[s])] = 1
        if delta < theta:
            break
    
    return policy, Q

UPDATES = 0

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    global UPDATES
    
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    
    # following p.80 of RL book
    while True:
        delta = 0
        for s in env.P:
            v_prev = V[s]
            v_updated = 0
            for a in env.P[s]:
                temp = 0
                for transition in env.P[s][a]:
                    prob, next_state, reward, done = transition
                    temp += prob * (reward + discount_factor * V[next_state])
                v_updated += policy[s,a] * temp
                UPDATES += 1
            V[s] = v_updated
            delta = max(delta, abs(v_prev - V[s]))
        if delta < theta:
            break
    
    return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        
        policy_stable = True
        for s in env.P:
            policy_prev = np.copy(policy[s])
            
            values = np.zeros(env.nA)
            for a in env.P[s]:
                for transition in env.P[s][a]:
                    prob, next_state, reward, done = transition
                    values[a] += prob * (reward + discount_factor * V[next_state])
            policy[s] *= 0
            policy[s, np.argmax(values)] = 1
            if sum(policy[s] != policy_prev) > 0:
                policy_stable = False
        if policy_stable:
            break
    
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        
        policy_stable = True
        for s in env.P:
            policy_prev = np.copy(policy[s])
            
            values = np.zeros(env.nA)
            for a in env.P[s]:
                for transition in env.P[s][a]:
                    prob, next_state, reward, done = transition
                    values[a] += prob * (reward + discount_factor * V[next_state])
            policy[s] *= 0
            policy[s, np.argmax(values)] = 1
            if sum(policy[s] != policy_prev) > 0:
                policy_stable = False
        if policy_stable:
            break
    
    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        
        policy_stable = True
        for s in env.P:
            policy_prev = np.copy(policy[s])
            
            values = np.zeros(env.nA)
            for a in env.P[s]:
                for transition in env.P[s][a]:
                    prob, next_state, reward, done = transition
                    values[a] += prob * (reward + discount_factor * V[next_state])
            policy[s] *= 0
            policy[s, np.argmax(values)] = 1
            if sum(policy[s] != policy_prev) > 0:
                policy_stable = False
        if policy_stable:
            break
    
    return policy, V

UPDATES = 0

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    global UPDATES
    
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    
    # following p.80 of RL book
    while True:
        delta = 0
        for s in env.P:
            v_prev = V[s]
            v_updated = 0
            for a in env.P[s]:
                temp = 0
                for transition in env.P[s][a]:
                    prob, next_state, reward, done = transition
                    temp += prob * (reward + discount_factor * V[next_state])
                v_updated += policy[s,a] * temp
            UPDATES += 1
            V[s] = v_updated
            delta = max(delta, abs(v_prev - V[s]))
        if delta < theta:
            break
    
    return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        V = policy_eval_v(policy, env, discount_factor)
        
        policy_stable = True
        for s in env.P:
            policy_prev = np.copy(policy[s])
            
            values = np.zeros(env.nA)
            for a in env.P[s]:
                for transition in env.P[s][a]:
                    prob, next_state, reward, done = transition
                    values[a] += prob * (reward + discount_factor * V[next_state])
            policy[s] *= 0
            policy[s, np.argmax(values)] = 1
            if sum(policy[s] != policy_prev) > 0:
                policy_stable = False
        if policy_stable:
            break
    
    return policy, V

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    global UPDATES
    
    # Start with an all 0 Q-value function and random policy
    Q = np.zeros((env.nS, env.nA))
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        delta = 0
        for s in env.P:
            for a in env.P[s]:
                q_prev = Q[s,a]
                q_updated = 0
                for transition in env.P[s][a]:
                    prob, s_next, reward, done = transition
                    q_next = np.zeros(env.nA)
                    for a_next in env.P[s_next]:
                        q_next[a_next] = Q[s_next, a_next]
                    # using update rule as written in Q2 of the homework
                    q_updated += prob * (reward + discount_factor * max(q_next))
                Q[s,a] = q_updated
                UPDATES += 1
                delta = max(delta, abs(q_prev - Q[s,a]))
            # learn policy implicitly
            policy[s] *= 0
            policy[s, np.argmax(Q[s])] = 1
        if delta < theta:
            break
    
    return policy, Q

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    global UPDATES
    
    # Start with an all 0 Q-value function and random policy
    Q = np.zeros((env.nS, env.nA))
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        delta = 0
        for s in env.P:
            for a in env.P[s]:
                q_prev = Q[s,a]
                q_updated = 0
                for transition in env.P[s][a]:
                    prob, s_next, reward, done = transition
                    q_next = np.zeros(env.nA)
                    for a_next in env.P[s_next]:
                        q_next[a_next] = Q[s_next, a_next]
                    # using update rule as written in Q2 of the homework
                    q_updated += prob * (reward + discount_factor * max(q_next))
                Q[s,a] = q_updated
                UPDATES += 1
                delta = max(delta, abs(q_prev - Q[s,a]))
            # learn policy implicitly
            policy[s] *= 0
            policy[s, np.argmax(Q[s])] = 1
        if delta < theta:
            break
    
    return policy, Q
