import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Callable
from enum import IntEnum
import random
from collections import defaultdict
import click

class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action):
    """
    Helper function to map action to changes in x and y coordinates

    Args:
        action (Action): taken action

    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]


def reset():
    """Return agent to start state"""
    return (0, 0)


# Q1
def simulate(state: Tuple[int, int], action: Action):
    """Simulate function for Four Rooms environment

    Implements the transition function p(next_state, reward | state, action).
    The general structure of this function is:
        1. If goal was reached, reset agent to start state
        2. Calculate the action taken from selected action (stochastic transition)
        3. Calculate the next state from the action taken (accounting for boundaries/walls)
        4. Calculate the reward

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))
        action (Action): selected action from current agent position (must be of type Action defined above)

    Returns:
        next_state (Tuple[int, int]): next agent position
        reward (float): reward for taking action in state
    """
    # Walls are listed for you
    # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
    walls = [
        (0, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 0),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 10),
        (6, 4),
        (7, 4),
        (9, 4),
        (10, 4),
    ]

    #check if goal was reached
    goal_state = (10, 10)
    start_state = (0,0)
    if state == goal_state: 
        next_state = reset() #if goal is reached reset to start
        reward = 1.0
        return next_state,reward

    # modify action_taken so that 10% of the time, the action_taken is perpendicular to action (there are 2 perpendicular actions for each action)
    action_taken = action
    if action_taken.name == 'UP' or action_taken.name == 'DOWN': #Perpendicular actions to UP,DOWN are LEFT, RIGHT
        choice_1 = [Action.LEFT,Action.RIGHT]
        action_taken = random.choice(choice_1) if round(np.random.random(),2) < 0.20 else action_taken
    elif action_taken.name == 'LEFT' or action_taken.name == 'RIGHT': #Perpendicular actions to LEFT,RIGHT are UP, DOWN
        choice_2 = [Action.UP,Action.DOWN]
        action_taken = random.choice(choice_2) if round(np.random.random(),2) < 0.20 else action_taken

    # calculate the next state and reward, given state and action_taken
    # You can use actions_to_dxdy() to calculate the next state
    # Check that the next state is within boundaries and is not a wall
    # One possible way to work with boundaries is to add a boundary wall around environment and
    # simply check whether the next state is a wall
    next_state = None
    reward = 0.0
    next_state = tuple(map(sum,zip(actions_to_dxdy(action_taken),state))) #Calculate position in matrix according to state and action taken
    next_state = state if any([(a < b) for a,b in zip(next_state,start_state)]) or any([(a >  b) for a,b in zip(next_state,goal_state)]) \
      or (next_state in walls) else next_state #Check if next state is outside boundaries or hitting walls

    return next_state, reward


# Q2
def manual_policy(state: Tuple[int, int]):
    """A manual policy that queries user for action and returns that action

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # Query user for each action
    usr_input = str(input("Please Enter an Action:")) 
    action_maps = {'LEFT':Action.LEFT,'RIGHT':Action.RIGHT,'UP':Action.UP,'DOWN':Action.DOWN} #Map User string Input to Action Object
    action = Action(action_maps[usr_input])
    return action
    pass


# Q2
def agent(
    steps: int = 100,
    trials: int = 1,
    policy=Callable[[Tuple[int, int]], Action],
):
    """
    An agent that provides actions to the environment (actions are determined by policy), and receives
    next_state and reward from the environment

    The general structure of this function is:
        1. Loop over the number of trials
        2. Loop over total number of steps
        3. While t < steps
            - Get action from policy
            - Take a step in the environment using simulate()
            - Keep track of the reward
        4. Compute cumulative reward of trial

    Args:
        steps (int): steps
        trials (int): trials
        policy: a function that represents the current policy. Agent follows policy for interacting with environment.
            (e.g. policy=manual_policy, policy=random_policy)

    """
    rewards_dict = defaultdict(list) #Maintain a rewards dictionary for all trials
    for t in range(trials):
        state = reset()
        cumulative_reward = 0.0
        i = 0
        while i < steps:
            # select action to take
            action = None
            reward = None
            action = policy(state) #Get Action Based on Policy
            # take step in environment using simulate()
            state,reward = simulate(state,action) #Get State and reward from environment
            # record the reward
            cumulative_reward += reward
            rewards_dict[t].append(cumulative_reward) #Store Cumulative Reward
            i+= 1
    return rewards_dict        

# Q3
def random_policy(state: Tuple[int, int]):
    """A random policy that returns an action uniformly at random

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    action_choice = [Action.LEFT,Action.RIGHT, Action.UP, Action.DOWN]
    return random.choice(action_choice) #Choose randomly from four actions (random uses uniform distribution)
    pass


# Q4
def worse_policy(state: Tuple[int, int]):
    """A policy that is worse than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    #As we have knowledge of environment, Intuitively Assign Higher Weight to going DOWN and LEFT 
    if round(np.random.random(),2) < 0.55:  
        action_choice = [Action.DOWN, Action.LEFT]
        return random.choice(action_choice)
    else:
        action_choice = [Action.UP, Action.RIGHT]
        return random.choice(action_choice)

    pass


# Q4
def better_policy(state: Tuple[int, int]):
    """A policy that is better than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    #Assign Higher Probability to the goal state direction
    if round(np.random.random(),2) < 0.70: 
        action_choice = [Action.UP, Action.RIGHT]
        return random.choice(action_choice)
    else:
        action_choice = [Action.DOWN, Action.LEFT]
        return random.choice(action_choice)
    pass

@click.command()
@click.option('--question',default='Q3',type = click.Choice(['Q2','Q3','Q4'],case_sensitive=False),
required=1,prompt='Which Question you want to run?',
help= 'Question to Run')
def main(question):
    """ Question : Q2: Manual Policy | Q3:Random Policy | Q4:Policy Comparisons """ 
    question_map = {'Q2':'manual_policy','Q3':'Random_Policy','Q4':'Policy Comparisons'}
    click.echo(f'Running Question {question} for "{question_map[question]}"')
    if question == 'Q2' :
        rewards_dict =  agent(100,1,manual_policy)
        print(rewards_dict)
    elif question == 'Q3':
        random_rewards= agent(10000,10,random_policy)
        random_rewards_df = pd.DataFrame(random_rewards)
        random_line = plt.plot(random_rewards_df.index,random_rewards_df,":")
        random_mean, = plt.plot(random_rewards_df.index,random_rewards_df.mean(axis=1),color="k",label="Random Policy",linewidth=2)
        plt.title("Random Policy")
        plt.xlabel("Steps")
        plt.ylabel("Cumulative Reward")
        plt.show()
    elif question == 'Q4':
        random_rewards= agent(10000,10,random_policy)
        random_rewards_df = pd.DataFrame(random_rewards)

        better_rewards= agent(10000,10,better_policy)
        better_rewards_df = pd.DataFrame(better_rewards)

        worse_rewards= agent(10000,10,worse_policy)
        worse_rewards_df = pd.DataFrame(worse_rewards)

        random_line = plt.plot(random_rewards_df.index,random_rewards_df,":")
        random_mean, = plt.plot(random_rewards_df.index,random_rewards_df.mean(axis=1),color="k",label="Random Policy",linewidth=2)
        better_line = plt.plot(better_rewards_df.index,better_rewards_df,":")
        better_mean, = plt.plot(better_rewards_df.index,better_rewards_df.mean(axis=1),color="b",label="Better Policy",linewidth=2)
        worse_line = plt.plot(worse_rewards_df.index,worse_rewards_df,":")
        worse_mean, = plt.plot(worse_rewards_df.index,worse_rewards_df.mean(axis=1),color="g",label="Worse Policy",linewidth=2)

        plt.legend(handles=[random_mean,better_mean,worse_mean])
        plt.title("Policy Comparisons")
        plt.xlabel("Steps")
        plt.ylabel("Cumulative Reward")
        plt.show()


if __name__ == "__main__":
    main()
