import torch
from torch import nn
import random

# Code structure based on lab4 from Reinforcement Learning course at University of Amsterdam


class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory)>self.capacity-1:
            del self.memory[0] # Would maybe be nice to store this for the case that memory.append fails
                               # but that requires quite extensive error handling which is not important here
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)