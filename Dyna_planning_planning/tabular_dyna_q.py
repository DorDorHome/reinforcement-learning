# implementation of 
# 1. random-sample one-step tabular Q-planning
# 2. Tabular Dyna-Q 
# 3. and Tabular Dyna-Q+

# instructions:
# in Dyna-Q, dynamics of the environment is not supposed to be known



# to-do:
# in random_sample_one_step_Q_planning, the reduction of alpha might need to tie to how many each state has been updated.



import gymnasium as gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

from collections import defaultdict
from lib import plotting

def random_sample_one_step_Q_planning(env, num_updates_for_each_state = None, randomize_state_action = False,  discount_factor = 1.0, alpha = 0.5, reduce_alpha_by_cycle = False,  convergence_criterion = 0.01 ):
    """
    input:
        env:  assumed a full model is available. 
            i.e. the env can take any combination of (state, action) as input
            and sample a next_state, reward 

        num_updates_for_each_state: 
            Number of updates to run for each state.
            If set to None, the algorithm will run until all possible (states, action) pairs have been updated once
            AND: convergence_criteria have been met for all of them:

        randomize_state_action:
            if set to true, state and acton will be selected randomly according to self.isd
                            and action will be selected randomly from env.nA
            Otherwise, a double loop will be used to make sure all (state, action ) pairs are visited.


        discount factor: for calculating discounted return.

        alpha: learning rate for the algorithm

        convergence_criteria:
            when num_updates is set to None,
            the algorithm will keep track of the last update deviation of all (state, action) pairs
            When they are all less than the convergence criteria, the algorithm will stop 


    output:
    ----------
    Q: an estimation of the Q value function for the optimal policy

    
    
    """
    # step 1: initialize a Q-value function:
        # A nested dictionary that maps state -> (action -> action-value).
    # 
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    state_space = np.arange(env.nS)



    # default state and acton, 
    # for when state-action is not randomized:
    state = 0 
    action = 0

    total_number_updates_for_all_states = num_updates_for_each_state*env.nS
    # set max number of update to infinity:
    if num_updates == None:
        num_updates = np.inf

    for t in itertools.count():
        if reduce_alpha_by_cycle:
            alpha_for_update = alpha /(t+1)
        else:
            alpha_for_update = alpha
        max_deviation = 0 # for each iteration, keep track of the maximum deviation between target and actual Q
        for (s, a)  in itertools.product(np.arange(env.nS), np.arange(env.nA)):
            # sample/loop through (state, action)
            if randomize_state_action:
                state = np.random.choice(np.arange(env.nS))
                action =np.random.choice(np.arange(env.nA))
            else:# loop to the next state, action:
                state = s
                action = a

            # set state of env:
            env.s = state
            # generate a next_state and reward:
            next_state, reward, done, _ = env.step(action)

            # update the Q function:
            best_action_value_on_next_state = np.max(Q[next_state])
            current_state_action_error = reward + discount_factor*best_action_value_on_next_state - Q[state][action]
            max_deviation = max( max_deviation, current_state_action_error )
            Q[state][action] +=  alpha_for_update*current_state_action_error


        # quit the loop if exceeds num_upates or if criteria is met
        if (max_deviation < convergence_criterion):
            print(f'algorithm stopped due to convergence. Error less than {convergence_criterion} for all state-action pairs')
            break
        if t > total_number_updates_for_all_states:
            print('algorithm stopped due to total number of updates exceeded with number of iteratons = {t}')
            break
        
        return Q
            

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def tabular_Dyna_Q(env, num_real_episode = 100, num_loop_in_between_real = 50, alpha = 0.5, discount_factor = 1.0, epsilon = 0.1 ):
    """
    implementation of Tabular Dyna-Q.
    Environment is assumed to be deterministic.
    

    input:
        env: OpenAI environment. Assumed 
        alpha: learning rate for the algorithm
        discount_factor
    output:
        Q: an estimate of the Q function under optimal policy.
        model: a model of the environment env, a nested dictionary that map state-action to (reward, next_state)
    
    """

    # initialize Q function:
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # model: nested dictionary state -> (action -> (reward, next_state) )
    model = defaultdict(lambda: defaultdict(lambda: (None, None)))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_real_episode),
        episode_rewards=np.zeros(num_real_episode))

    # epsilon-greedy policy:
    epsilon_greedy_policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for epi in itertools.count(): # keep track of the number of episode
        # Print out which episode we're on, useful for debugging.
        if (epi+ 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(epi + 1, num_episodes), end="")
            sys.stdout.flush()
        state = env.reset()
        for step in itertools.count():
            action_probs = epsilon_greedy_policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
                        
            # Update statistics
            stats.episode_rewards[epi] += reward
            stats.episode_lengths[epi] = t

            # 1-step TD update using real experience:
            td_target = reward + discount_factor*np.max(Q[next_state])
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            # update model of the env:
            model[state][action] = (reward, next_state)

            # update by looping through the model:
            ## note: don't overwrite next_state!
            for it_using_model in range(num_loop_in_between_real):
                # get state that appeared from model:
                mock_state = np.random.choice(list(model.keys()))
                # get action previous taken:
                mock_action = np.random.choice(model[state].keys())
                model_reward, model_next_state = model[mock_state][mock_action]
                # update:
                td_target = model_reward + discount_factor*np.max(Q[model_next_state])
                td_delta = td_target - Q[mock_state][mock_action]
                Q[mock_state][mock_action] += alpha * td_delta


                


            state = next_state

            if done:
                break# quit this episode, go to the next one
    # condition for quiting all episode:
        if epi > num_real_episode:
            break
    
    return Q, stats, model
    
    

def tabular_Dyna_Q_plus():
    pass