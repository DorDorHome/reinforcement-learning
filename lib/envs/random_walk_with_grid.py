## Created by Alvin, Jan 13, 2024:

# this environment is created to replicate the 
## Random walk in example 6.2 of the book
#Reinforcemen Learning: An introduction

# to-do-list:
# to give the options of using list/sets in pos/negative terminal position
# the super.__init__ in RandomWalk might override (and thus make unnecessary) the definition of
# self.P, self.nS and self.nA

import io
# import gym
import gymnasium as gym
import numpy as np
import sys

try:
    from . import discrete
except:
    import discrete

STAY = 0
RIGHT = 1
LEFT = 2
UP = 3
DOWN = 4




class RandomWalk(discrete.DiscreteEnv):
    """
    Creating a grid world where the transition depends on action plus a random force
    When action is not to move at all, the result would be similar to random walk in physics.

    by default (when terminal_states is None), the terminal states are set to be the
    top left (negative values) and the bottom right (positive values).

    When reaching an edge, if the combined action of agent or the environment moves the agent towards outside 
    
    explanation of selected variables:

    reward_grid: a grid to indicate that what rewards will be stepping into the cell in the grid


    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape = [1, 7], positive_terminal_pos = None, negative_terminal_pos = None,
                 reward_into_pos_term_states = 1,
                 reward_into_neg_term_states = -1, reward_grid = None,  drift_vector_prob = None, initial_state_distribution = None):
        
        self.space_is_1d = None

        # 1. defines a grid-like world for random walk to "walk" on
        # if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
        #     raise ValueError('shape argument must be a list/tuple of length 2')
        self.shape = shape 
        

        # 2.
        # calculate and save the number of states:
        nS = np.prod(shape)

        # 3. number of actions in each state (except terminal states)
        ## the number of actions depends on whether
        ## the grid is 1d or 2d
        ## if 1d, the number of actions is 3 (left, right, stay)
         
        if len(shape) == 2 and (shape[0] ==1 or shape[1]==1):
            self.space_is_1d = True
            nA = 3
            print('space is 1d')
            
        elif len(shape) ==2:
            # covering the case where at least one dimension in shape is greater than one:
            self.space_is_1d = False
            print('space is 2d')
            nA =5

        else:
            # covering l
            raise ValueError('please check shape, only support 2d grid. For 1d grid,\
                              please enter as 2d grid with one dimension set to 1')
        
        self.action_to_action_in_array = {}
        if self.space_is_1d:
            self.action_to_action_in_array[STAY] = np.array([ 0, 0])
            self.action_to_action_in_array[RIGHT] = np.array([0, 1])
            self.action_to_action_in_array[LEFT] = np.array([0,-1])
        else:
            self.action_to_action_in_array[STAY] = np.array([ 0, 0])
            self.action_to_action_in_array[RIGHT] = np.array([0, 1])
            self.action_to_action_in_array[LEFT] = np.array([0,-1])
            self.action_to_action_in_array[UP] = np.array([-1, 0])
            self.action_to_action_in_array[DOWN] = np.array([1,0])

            
        ## save all available action in array form:
        self.action_arrays = [ action_array for action_array in self.action_to_action_in_array.values() ]



        # the height of the space:
        MAX_Y = shape[0] 
        # the width of the space:
        MAX_x = shape[1]

        if drift_vector_prob is not None :
            # 1. check the entered drift_vector_prob adds up to one:
            check_drift_prob_sum = 0
            for prob in drift_vector_prob.values():
                check_drift_prob_sum += prob
            assert check_drift_prob_sum == 1
            # 2. check the possible drift direction is of shape (2,) and has integer values
            for vec in drift_vector_prob:
                assert vec.shape == (2,)
                assert isinstance(vec[0], np.int64) 
                assert isinstance(vec[1], np.int64)

            # 3. check the number of items matches the number of available actions:
            assert len(self.action_arrays) == len(drift_vector_prob.keys())


            # save it as an attribute:
            self.drift_vector_prob = drift_vector_prob

        elif drift_vector_prob is None: 
        # set drift vector prob to be equal probability if not set
            if self.space_is_1d:
                self.drift_vector_prob = {
                    (0,1): 0.5, #RIGHT
                    (0, -1): 0.5, #LEFT
                    (0, 0): 0  #STAY
                }
            else :
                self.drift_vector_prob = {

                    (0,1): 0.25, #RIGHT
                    (0, -1): 0.25, #LEFT
                    (1, 0): 0.25,#DOWN
                    (-1, 0): 0.25,#UP
                    (0, 0): 0  #STAY
                }




        print('drift vector prob:', self.drift_vector_prob)
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
        self.grid = grid
        
        if positive_terminal_pos is None:# define default positive terminal states position:
            # bottom right is positive
            positive_terminal_pos = (self.grid.shape[0] -1 , self.grid.shape[1] -1)
        if negative_terminal_pos is None: # likewise for negative terminal states:
            # top left is negative:
            negative_terminal_pos = (0,0)

        self.positive_terminal_pos = positive_terminal_pos
        self.negative_terminal_pos = negative_terminal_pos
        print(negative_terminal_pos)

        print(positive_terminal_pos)



        assert isinstance(positive_terminal_pos, tuple), 'please enter a numpy array as terminal states'

        # check the positive terminal states is inside the grid


        assert isinstance(negative_terminal_pos, tuple), 'please enter a numpy array as terminal states'
        # check for no overlap:
        # overlap = np.logical_and(np.array(positive_terminal_pos), np.array(negative_terminal_pos))
        # assert not overlap.any(), "positive and neg term states overlapped!"

        if isinstance(positive_terminal_pos, list):
            self.terminal_pos = positive_terminal_pos+ negative_terminal_pos
        elif isinstance(positive_terminal_pos, tuple):
            self.terminal_pos = [positive_terminal_pos, negative_terminal_pos]
        else:
            raise TypeError

        print('terminal_pos is', self.terminal_pos)
        # next, define the reward of stepping into a cell:

        if reward_grid == None:
            # default is, all rewards are zero, except when specified otherwise
            # and except in the designated pos and neg terminal pos
            self.reward_grid = np.zeros(self.grid.shape)
        else:
            assert reward_grid.dtype == float, 'rewards can only be floats! '
            assert reward_grid.shape == shape, 'the shape of reward'
            self.reward_grid = np.array(reward_grid)

        # at positive terminal pos:
        self.reward_grid[positive_terminal_pos] = reward_into_pos_term_states

        # at negative terminal pos:
        self.reward_grid[negative_terminal_pos] = reward_into_neg_term_states

        



        # not used:
        # # set location of terminal state:
        # terminal_reward = np.zeros(shape)
        # if positive_terminal_pos == None:
        #     # set default terminal states to be the bottom right
        #     terminal_reward[shape[0]-1, shape[1]-1] = reward_into_pos_term_states
        # else:
        #     terminal_reward[positive_terminal_pos] = reward_into_pos_term_states

        
        # if negative_terminal_pos == None:
        #     # set default terminal states to be the top left:
        #     terminal_reward[0,0] = reward_into_neg_term_states
        # else:
        #     terminal_reward[negative_terminal_pos] =reward_into_neg_term_states

        # print(f'here is the world, with {reward_into_pos_term_states} indicating positive terminal states.\
        #       and {reward_into_neg_term_states} indicating negative terminal states: ')
        # # print_board(terminal_reward, reward_into_pos_term_states = reward_into_pos_term_states, reward_into_neg_term_states= reward_into_neg_term_states)
        # self.terminal_reward = terminal_reward

        # set up reset state distribution:
        if initial_state_distribution == None:
            isd = np.zeros(shape)
            print('initializing position is in middle')
            start_y = shape[0]//2
            start_x = shape[1]//2
            # this defines the initial state distribution:
            isd[start_y, start_x] =1


        else:
            isd = np.array(initial_state_distribution)
            # check this is a probability distribution:
            assert isd.sum() ==1, 'Please enter a probability distribution! Prob should add up to one.'
        print('probability disbribution for starting state: ', isd)





        # set up the transition probabilities:



        it = np.nditer(grid, flags = ['multi_index'])

        while not it.finished:
            s = it.iterindex # get the state
            y, x = it.multi_index # get the coordinates of the state
            pos_np = np.array([y, x])
            #y is vertical dimension and x is horizontal
            print('state is ', s)
            print('pos is ', y ,x)
            # each state allows equal number of actions:
            P[s] = {a: [ ] for a in range(nA) }

            # to check whether s is a terminal state:
            is_done = (y, x) in self.terminal_pos  #bool(terminal_reward[y][x])
            print('is done:', is_done)
            if is_done:
                # set all actions to have no effect (next_state = s, and reward = 0), with probability one
                
                for a in range(nA):
                    P[s][a] = [(1.0, s, 0, True)]
                # P[s][UP] = [(1.0, s, 0, True)]
                # P[s][DOWN] = [(1.0, s,  0, True)]
                # P[s][RIGHT] = [(1.0, s,  0, True)]
                # P[s][LEFT] = [(1.0, s,  0, True)]
                # P[s][STAY] = [(1.0, s,  0, True)]

            # the reward and whether a state is 
            else:
                for a in range(nA):
                    P[s][a]= self._get_transition_prob_from_combined_effect(pos_np, a,self.drift_vector_prob )
                # P[s][UP] = self._get_transition_prob_from_combined_effect(pos_np, UP,self.drift_vector_prob )
                # # print('initial position: [',  y, x, ']' )
                # # print('action: up')
                # # print('transition probability: ', P[s][UP] )
                # P[s][DOWN] = self._get_transition_prob_from_combined_effect(pos_np,DOWN, self.drift_vector_prob )
                # # print('initial position: [',  y, x, ']' )
                # # print('action: DOWN')
                # # print('transition probability: ', P[s][DOWN] )
                
                # P[s][RIGHT] = self._get_transition_prob_from_combined_effect(pos_np, RIGHT,self.drift_vector_prob )
                # # print('initial position: [',  y, x, ']' )
                # # print('action: RIGHT')
                # # print('transition probability: ', P[s][RIGHT] )
                
                # P[s][LEFT] = self._get_transition_prob_from_combined_effect(pos_np, LEFT,self.drift_vector_prob )
                
                # # print('initial position: [',  y, x, ']' )
                # # print('action: LEFT')
                # # print('transition probability: ', P[s][LEFT] )
                
                # P[s][STAY]  = self._get_transition_prob_from_combined_effect(pos_np, STAY,self.drift_vector_prob )

                # # print('initial position: [',  y, x, ']' )
                # # print('action: STAY')
                # # print('transition probability: ', P[s][STAY] )
            it.iternext()


                   # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm 
        self.P = P
        self.reward_into_pos_term_states = reward_into_pos_term_states
        self.reward_into_neg_term_states = reward_into_neg_term_states
        super(RandomWalk, self).__init__(nS, nA, P, isd)
            





    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord
    
    def _get_transition_prob_from_combined_effect(self, current_pos, action, drift_prob):
        """
        current_pos: 
        current position, in index coordinate( going down is positive), as np.array in int64

        action: 
        another np.array in int64
        
        drift_prob:
        the effect of drifting, as a dictionary
            keys: drift direction (tuple), in int64
            values: probability of that drift direction.     
        """

        next_pos_to_prob = {}
        transition_list = []
        print('starting state position: ', current_pos)
        print('start calculating transition prob....')
 
        for drift in drift_prob:
            # next position is the combined effect of current position and drift:
            # print('before limiting coord: ',current_pos +self.action_to_action_in_array[action]+np.array(drift) )
        
            
            next_pos = tuple(self._limit_coordinates(current_pos +self.action_to_action_in_array[action]+np.array(drift)).astype(int))
            # accumlate the pobability 
            # print('next pos:', next_pos)
            if next_pos in next_pos_to_prob:
                # note that there, next_pos is a tuple
                next_pos_to_prob[next_pos] +=drift_prob[drift] # drift is also a tuple
            else:
                next_pos_to_prob[next_pos] = drift_prob[drift]
            
        # finally, get a list of all (probability, state, reward, done):
        for next_pos in next_pos_to_prob:
            reward = self.reward_grid[next_pos[0], next_pos[1]]
            #print(f'reward from {current_pos} to {next_pos} is {reward}')
            done = bool(reward)
            next_pos_in_s = self.grid[next_pos[0], next_pos[1]]
            transition_list.append((next_pos_to_prob[next_pos], next_pos_in_s, reward, done ))
        print('\n')
        return transition_list
        
    def render(self, mode ='human', close = False):
        self._render(mode, close)

    def _render(self, mode = 'human', close = False):
        """ Renders the current grid layout for random walk

         For example, a 4x4 grid with the mode="human" looks like:
            L  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  W
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile=io.StringIO() if mode == 'ansi' else sys.stdout

        # put the state space in a grid
        grid = np.arange(self.nS).reshape(self.shape)

        it = np.nditer(grid, flags =['multi_index'])
        while not it.finished:
            s = it.iterindex
        
            # the position of the state:
            y, x = it.multi_index 

            # position of the current state, marked as x
            if self.s == s:
                output = " x "
            # mark the terminal states with pos reward
            elif (y,x) == self.positive_terminal_pos :
                output = f"{self.reward_into_pos_term_states}"
            # mark the terminal states with neg reward
            elif (y,x) == self.negative_terminal_pos:
                output = f"{self.reward_into_neg_term_states}"
            else:
                output = " O "

            # when the position of the state is at the far left:
            if x == 0:
                output = output.lstrip()
            # when the position of the state is at the far right:
            if x == self.shape[1] - 1:
                output = output.rstrip()
                # next line
                output += "\n" 


            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()







# abandoned: use self.render instead for better generalizations
# helper function for printing game board:

        
# def print_board(board, reward_into_pos_term_states,  reward_into_neg_term_states):
#     print('ugly print of board', board)
#     chars = { reward_into_neg_term_states: 'lose', 0: '0', reward_into_pos_term_states: 'win' }
#     hline = '-' * (board.shape[1] * 4 - 1)
    
#     for i, row in enumerate(board):
#         # Print the row with vertical separators
#         print(' | '.join(chars[val] for val in row))
#         # Print horizontal line separator after each row except the last
#         if i < board.shape[0] - 1:
#             print(hline)


