## Created by Alvin, Jan 13, 2024:

# this environment is created to replicate the 
## Random walk in example 6.2 of the book
#Reinforcemen Learning: An introduction

import io
# import gym
import gymnasium as gym
import numpy as np
import sys

from . import discrete

class RandomWalk(discrete.DiscreteEnv):
    """
    Creating a grid world with 
    
    
    
    
    
    
    
    """




    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape = [1, 7]):
        # 1. defines a grid-like world for random walk to "walk" on
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')
        self.shape = shape 


        # 2. the number of states:
        nS = np.prod(shape)

        # 3. number of actions in each state (except terminal states)
        ## the number of actions depends on whether
        ## the grid is 1d or 2d
        ## if 1d, the number of actions is 2
        nA = 2**len(len(shape))

        





