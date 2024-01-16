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

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
STAY = 4

class RandomWalk(discrete.DiscreteEnv):
    """
    Creating a grid world where the transition depends on action plus a random force
    When action is not to move at all, the result would be similar to random walk in physics.

    by default (when terminal_states is None), the terminal states are set to be the
    top left (negative values) and the bottom right (positive values).

    When reaching an edge, if the combined action of agent or the environment moves the agent towards outside 
    
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape = [1, 7], positive_terminal_states = None,
                 negative_terminal_states = None, reward_into_pos_term_states = 1,
                 reward_into_neg_term_states = -1, drift_vector_prob = None):
        


        # 1. defines a grid-like world for random walk to "walk" on
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')
        self.shape = shape 


        # 2. the number of states:
        nS = np.prod(shape)

        # 3. number of actions in each state (except terminal states)
        ## the number of actions depends on whether
        ## the grid is 1d or 2d
        ## if 1d, the number of actions is 3 (left, right, stay)
        nA = 2**len(len(shape)) + 1

        # the height of the space:
        MAX_Y = shape[0] 
        # the width of the space:
        MAX_x = shape[1]
        if drift_vector_prob is not None :
            # check the entered drift_vector_prob adds up to one:
            check_drift_prob_sum = 0
            for prob in drift_vector_prob.values():
                check_drift_prob_sum += prob
            assert check_drift_prob_sum == 1
            # check the possible drift direction is of shape (2,) and has integer values
            for vec in drift_vector_prob:
                assert vec.shape == (2,)
                assert isinstance(vec[0], np.int64) 
                assert isinstance(vec[1], np.int64)
            self.drift_vector_prob = drift_vector_prob
        else:
            self.drift_vector_prob = {

                np.array([0,1]): 0.25, 
                np.array([0, -1]): 0.25,
                np.array([1, 0]): 0.25,
                np.array([-1, 0]): 0.25,
                np.array([0, 0]): 0
            }

    
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

        # check
        assert isinstance(positive_terminal_states, np.ndarray), 'please enter a numpy array as terminal states'
        assert isinstance(negative_terminal_states, np.ndarray), 'please enter a numpy array as terminal states'
        # check terminal shape and grid shape the same:
        assert grid.shape == positive_terminal_states.shape, " terminal state shape needs to be the same as grid shape!"
        assert grid.shape == negative_terminal_states.shape, " terminal state shape needs to be the same as grid shape!"
        overlap = np.logical_and(positive_terminal_states, negative_terminal_states)
        assert not overlap.any(), "positive and neg term states overlapped!"


        # set location of terminal state:
        terminal = np.zeros(shape, dtype =np.bool)
        if positive_terminal_states == None:
            # set default terminal states to be the top left
            terminal[0][0] = 1
        else:
            terminal[positive_terminal_states] = True

        
        if negative_terminal_states == None:
            # set default terminal states to be the bottom right:
            terminal[shape[0]-1, shape[1]-1] = True

        print('here is the world, with 1 indicating positive terminal states.\
              and -1 indicating negative terminal states: ')
        print()

        # set up the transition probabilities:
        it = np.nditer(grid, flags = ['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index # y is vertical dimension

            # each state allows equal number of actions:
            P[s] = {a: [ ] for a in range(nA) }

            # to check whether s is a terminal state:
            is_done = terminal[y][x]

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, 0, True)]
                P[s][RIGHT] = [(1.0, s, 0, True)]
                P[s][DOWN] = [(1.0, s, 0, True)]
                P[s][LEFT] = [(1.0, s, 0, True)]

            # if s is not a terminal state
            else:

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord
    
    def check_positive_term(pos):




    
    def check_negative_term(pos): 

    
    def _calculate_transition_prob(self, curren_position ,action, drift_vector_prob):
        """
        Given a combined effect of action plus the random drift,
        get the (prob, new_state, reward, is_done) lists for the next state.
        ___________
        input:
        current_position: 
            numpy array of shape (2,), with integer values, y and x coordinates using index convention 
        action:
            numpy array of shape (2,), only five possibilities:
            [1,0], [-1,0], [0, 1], [0, -1], [0,0] 
        drift_vector_prob:
        contains the probability of each drift direction, among the five possibilities:
        [1,0], [-1,0], [0, 1], [0, -1], [0,0] 


        """

        next_pos_to_prob_reward_dict 
        for drift_vec in drift_vector_prob:







    















