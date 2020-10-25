import torch
from torch import nn

# Code structure based on lab4 from Reinforcement Learning course at University of Amsterdam



def compute_q_vals(Q, states, actions=None):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    all_actions = Q(states)
    if not actions is None:
        # So far only compatible with a single action dimension per timestep
        Q_values = all_actions[range(all_actions.shape[0]), actions.squeeze().tolist()].unsqueeze(dim=1)
    else: # If actions are not defined, we take the best action's Q-value
        Q_values, _ = all_actions.max(dim=1, keepdim=True)
        # Could be updated to include the argmax as well
    return Q_values
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    # YOUR CODE HERE
    
    # First, we need to find the max Q-value (over actions) from the next states
    future_Q_vals = compute_q_vals(Q, next_states)
    
    # if the next state is terminal, the Q-value should be zero:
    # turns out that 'dones' is not actually a boolean tensor, but an integer tensor. Waste of memory..
    # In codegrade, the dones are in fact boolean tensors. What a mess :')
    if dones.dtype == torch.bool: # Use boolean operators when actual boolean tensor
        done_tensor = ~dones
        future_Q_vals *= done_tensor
    else: # Must be a numerical tensor representing boolean values then
        done_tensor = 1 - dones  
        future_Q_vals *= done_tensor
    
    # With some complicated indexing tricks we could prevent the done states from passing through the Q-net
    # but this will likely not save a significant amount of processing time
    
    target = rewards + discount_factor * future_Q_vals
    
    return target


def train(Q, memory, optimizer, batch_size, discount_factor, target_network):

    # don't learn without some decent experience
    if len(memory) < batch_size and memory.capacity >= batch_size:
        return None, None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(min(batch_size, memory.capacity))
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(target_network, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = nn.functional.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), q_val.abs().max().item()  # Returns a Python scalar, and releases history (similar to .detach())
