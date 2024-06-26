import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()

def plot_value_function_of_grid_world(V, title = 'Value Function over grid space', shape = None):
    """
    plots the value function using V, and given the shape of the grid world
    
    V is over state space, { 0, 1, 2, .....}
    shape: a tuple. The space is structured in this way: grid_world = np.arange(env.nS).reshape(shape)

    """
    if shape is None:
        raise ValueError("Shape parameter is required to visualize the value function.")

    # Initialize a grid of the specified shape with zeros
    grid = np.zeros(shape)
    
    # Fill the grid with values from V
    for state, value in V.items():
        # Calculate the position in the grid from the state index
        row, col = divmod(state, shape[1])
        grid[row, col] = value

    # Create a figure and a 3D subplot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create a meshgrid to align the bars
    xpos, ypos = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    # Bar dimensions
    dx = dy = 0.8
    dz = grid.flatten()

    # Plot
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_zlabel('Value')

    # Set the ticks for the x and y axes
    ax.set_xticks(np.arange(shape[1]) + dx/2)
    ax.set_xticklabels(np.arange(shape[1]))
    ax.set_yticks(np.arange(shape[0]) + dy/2)
    ax.set_yticklabels(np.arange(shape[0]))

    plt.show()
    # min_x = 0

    # max_x = shape[1] -1
    # min_y = 0
    # max_y = shape[0] -1

    # x_range = np.arange(min_x, max_x + 1)
    # y_range = np.arange(min_y, max_y + 1)
    # X, Y = np.meshgrid(x_range, y_range)

    # Z = np.apply_along_axis(lambda _: V[ _[0]*max_x+ _[1]], 2, np.dstack([X, Y]))

    # def plot_surface(X, Y, Z, title):
    #     fig = plt.figure(figsize=(20, 10))
    #     ax = fig.add_subplot(111, projection='3d')
    #     surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    #                            cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    #     ax.set_xlabel('x-direction')
    #     ax.set_ylabel('y-direction')
    #     ax.set_zlabel('Value')
    #     ax.set_title(title)
    #     ax.view_init(ax.elev, -120)
    #     ax.invert_yaxis() 
    #     fig.colorbar(surf)
    #     plt.show()

    # def plot_3d_barchart(X, Y, Z, title):
    #     fig = plt.figure(figsize=(20, 10))
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     # Size of the bars
    #     dx = dy = 0.75  # Width and depth of the bars
    #     dz = Z.flatten()  # Heights of the bars
        
    #     # Coordinates for the bars
    #     xpos = X.flatten()
    #     ypos = Y.flatten()
    #     zpos = np.zeros_like(dz)  # All bars starting at z=0
        
    #     # Plotting the bars
    #     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, cmap=matplotlib.cm.coolwarm)
        
    #     # Labels and title
    #     ax.set_xlabel('x-direction')
    #     ax.set_ylabel('y-direction')
    #     ax.set_zlabel('Value')
    #     ax.set_title(title)
        
    #     # Inverting y axis
    #     ax.invert_yaxis()
        
    #     plt.show()
    
    # plot_3d_barchart(X, Y, Z, f'{title}')
    # # plot_surface(X, Y, Z, f'{title}')



def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    
    # for key in V.keys():
    #     print(key)
    #     print(type(key))
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))



def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3

def plot_episode_two_stats(stats1, stats2, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats1.episode_lengths, color='blue', label='stat1')
    plt.plot(stats2.episode_lengths, color='red', label='stat2')
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed1 = pd.Series(stats1.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed2 = pd.Series(stats2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    
    plt.plot(rewards_smoothed1, color ='blue' )
    plt.plot(rewards_smoothed2, color = 'red' )
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats1.episode_lengths), np.arange(len(stats1.episode_lengths)), color ='blue' )
    plt.plot(np.cumsum(stats2.episode_lengths), np.arange(len(stats2.episode_lengths)), color ='red' )
       
    
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3
