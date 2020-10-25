import torch
from torch import nn

# Code structure based on lab4 from Reinforcement Learning course at University of Amsterdam


class QNetwork(nn.Module):
    
    def __init__(self, in_size, out_size, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(in_size, num_hidden)
        self.l2 = nn.Linear(num_hidden, out_size)
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x):

        x = torch.tensor(x)  # Seems like this does nothing, even when numpy gets passed into Q
        x = x.view(-1, self.in_size)

        layer_1 = self.l1(x)
        hidden = nn.functional.relu(layer_1) # Apply activation function as function rather than later
        output = self.l2(hidden).view(-1, self.out_size)

        return output


