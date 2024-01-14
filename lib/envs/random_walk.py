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
    Creating a grid world where the transition does not depend on action
    similar to random walk in physics.

    inputs: 
    
    
    
    
    
    
    
    
    """




    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape = [1, 7], terminal_states = None):
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

        # the height of the space:
        MAX_Y = shape[0] 
        # the width of the space:
        MAX_x = shape[1]

        # create a dictionary to contain the transition prob
        # this is a nested dictionary
        # expected to contain, for each state s
        #  a dictionary P[s] , with action as key
        # for each action a, P[s][a] gives a list of 
        # (transition_prob, next_state, reward, done)
        P = {}

        # create the states according to the shape given (and thus nA)
        # and store them in an array to give it a natural geometric structure:
        grid = np.arange(nS).reshape(shape)

        # set location of terminal state:
        terminal = np.zeros(shape, dtype =np.bool)
        if terminal_states == None:
            # set default terminal states to be the left most


        else:
            assert isinstance(terminal_states, np.ndarray), 'please enter a numpy array as terminal states'
            assert

        # check terminal shape and grid shape the same:
        assert grid.shape == terminal.shape, ""

        it = np.nditer(grid, flags = ['multi_index'])

        while not it.finished:









