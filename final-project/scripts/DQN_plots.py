import numpy as np
import matplotlib.pyplot as plt

# Code structure based on lab4 from Reinforcement Learning course at University of Amsterdam


def plot_smooth(x, N, show=False):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    plt.plot((cumsum[N:] - cumsum[:-N]) / float(N))

    if show:
        plt.show()
