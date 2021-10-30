import gym
from typing import Optional
from collections import defaultdict
from gym.core import ActionWrapper
import numpy as np
from typing import List, Dict, Optional,Sequence,Callable,Tuple
import sys

def generate_episode(env: gym.Env,Q:defaultdict,epsilon:float,num_steps:int) -> List[Tuple[int,int,float]]:
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        Q (defaultdict): Q-values
        epsilon (float): epsilon for epsilon greedy
        num_steps (int): Number of steps
    """
    episode = []
    state = env.reset()
    i = 0
    while i < num_steps:
        action = epsilon_greedy(Q,state,epsilon)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            return episode,done
        state = next_state
        i+=1
    return episode,done

#on-policy Monte Carlo control(for epsilon-soft policies)
def on_policy_mc_control_epsilon_soft(
    env: gym.Env, num_steps: int, gamma: float, epsilon: float
):
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    episodes_completed = []
    ep = 0
    steps = 0
    ep_count = 0
    while steps < num_steps:
        G = 0.0
        episode,done= generate_episode(env,Q,epsilon,num_steps)
        ep = len(episode)
        if done:
            ep_count+=1
        if steps + ep > num_steps:
            episodes_completed[steps:num_steps+1] = [ep_count]*(num_steps-steps)
            break
        else:
            episodes_completed.extend([ep_count]*ep)
        for t in range(len(episode)-1,-1,-1):
        # For each episode calculate the return
        # Update Q and N
            reward = episode[t][2]
            G = gamma * G + reward
            current_state = episode[t][0]
            current_action = episode[t][1]
            previous_state_action = [(i[0],i[1]) for i in episode[0:t]]
            if (current_state,current_action) not in previous_state_action:
                N[current_state][current_action]+=1.0
                Q[current_state][current_action]+= (1/N[current_state][current_action]) * (G-Q[current_state][current_action])
        steps+=ep
    return Q, episodes_completed


def argmax(arr: Sequence[float]) -> int:
    """Argmax that breaks ties randomly

    Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

    Args:
        arr: sequence of values
    """
    return np.random.choice(np.flatnonzero(arr == arr.max()))

# epsilon greedy action selection
def epsilon_greedy(Q,state,epsilon) -> int:
    """Epsilon-greedy action selection
    Given a Q function and a state, returns an action selected by epsilon-greedy exploration.

    Args:
        Q (defaultdict): Q-values
        state (int): state
        epsilon (float): epsilon for epsilon greedy
    """
    num_actions = len(Q[state])
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)
    else:
        return argmax(Q[state])

#on-policy TD control
def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float):
    """One-step SARSA
    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    episodes_completed = []
    ep = 0
    state = env.reset()
    action = epsilon_greedy(Q, state, epsilon)
    for step in range(num_steps):
        next_state, reward, done, _ = env.step(action)
        next_action  = epsilon_greedy(Q, next_state, epsilon)
        if done:
            ep+=1
            state = env.reset()
            action = epsilon_greedy(Q, state, epsilon)
        else:
            Q[state][action] += step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action
        episodes_completed.append(ep)
    return Q, episodes_completed  
        
def nstep_sarsa(
    env: gym.Env,
    num_steps: int,
    n: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """N-step SARSA
    On-policy TD Control to find optimal epsilon-greedy policy

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    ep = 0
    episodes_completed = []
    tau = 0
    t_episode = 0
    previous_actions = []
    previous_rewards = []
    previous_states = []
    state = env.reset()
    action = epsilon_greedy(Q, state, epsilon)
    previous_actions.append(action)
    previous_states.append(state)    
    for steps in range(num_steps):
        next_state, reward, done, _ = env.step(action)
        next_action  = epsilon_greedy(Q,next_state, epsilon)
        previous_rewards.append(reward)
        previous_states.append(next_state)
        previous_actions.append(next_action)
        if done:
            ep+=1
            t_episode = 0
            previous_actions = []
            previous_rewards = []
            previous_states = []
            state = env.reset()
            action = epsilon_greedy(Q, state, epsilon)
            previous_actions.append(action)
            previous_states.append(state)
        else:
            tau = t_episode - n + 1
            if tau >= 0:
                G = 0.0
                G = np.sum([np.power(gamma,(i-tau-1))*previous_rewards[i] for i in range(tau+1,min(tau+n,num_steps))])
                if tau + n < num_steps:
                    G +=np.power(gamma,n) * Q[previous_states[(tau+n)]][previous_actions[(tau+n)]]
                tau_s, tau_a = previous_states[tau], previous_actions[tau]
                Q[tau_s][tau_a] += step_size * (G - Q[tau_s][tau_a])
            t_episode += 1
            state = next_state
            action = next_action
        episodes_completed.append(ep)
    return Q, episodes_completed


def calculate_expectation(Q,state,epsilon) -> float:
    """Calculate the expectation of the Q-values for a given state.

    Args:
        Q (defaultdict): Q-values
        state (int): state
        epsilon (float): epsilon for epsilon greedy
    """
    exp = 0
    num_actions = len(Q[state])
    best_action = argmax(Q[state])
    probs = [epsilon/num_actions]*num_actions
    probs[best_action] = 1-epsilon + (epsilon/num_actions)
    for i in range(num_actions):
        exp+= probs[i]*Q[state][i]
    return exp

def exp_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    episodes_completed = []
    ep = 0
    state = env.reset()
    for step in range(num_steps):
        action  = epsilon_greedy(Q, state, epsilon)
        next_state, reward, done, _ = env.step(action)
        if done:
            ep+=1
            state = env.reset()
        else:
            Q[state][action] += step_size * (reward + gamma * calculate_expectation(Q, next_state, epsilon) - Q[state][action])
            state = next_state
        episodes_completed.append(ep)
    return Q, episodes_completed        



def q_learning(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    episodes_completed = []
    ep = 0
    state = env.reset()
    for step in range(num_steps):
        action  = epsilon_greedy(Q, state, epsilon)
        next_state, reward, done, _ = env.step(action)
        if done:
            ep+=1
            state = env.reset()
        else:
            best_action = argmax(Q[next_state])
            Q[state][action] += step_size * (reward + gamma * Q[next_state][best_action] - Q[state][action])
            state = next_state
        episodes_completed.append(ep)
    return Q, episodes_completed        

def td_prediction(env: gym.Env, gamma: float, step_size:float, episodes, n:int) -> defaultdict:
    """TD Prediction

    This generic function performs TD prediction for any n >= 1. TD(0) corresponds to n=1.

    Args:
        env (gym.Env): a Gym API compatible environment
        gamma (float): Discount factor of MDP
        step_size (float): step size
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
    """
    v = defaultdict(float)
    for episode in episodes:
        t = 0
        tau = 0
        T = sys.maxsize
        while True:
            if t < T:
                if t+1 == len(episode):
                    T = t+1
            tau = t - n + 1
            if tau >= 0:
                G = sum([np.power(gamma,(i-tau))*episode[i][2] for i in range(tau,min(tau+n,T))])
                if tau + n < T:
                    state_tpn = episode[tau+n][0]
                    G+= np.power(gamma,n) * v[state_tpn]
                state_tau = episode[tau][0]
                v[state_tau] += step_size * (G - v[state_tau])        
            t += 1
            if tau == T - 1:
                break
    return v


def mc_prediction(
    episodes,
    gamma: float,
) -> defaultdict:
    """On-policy Monte Carlo policy evaluation. First visits will be used.

    Args:
        gamma (float): Discount factor of MDP

    Returns:
        V (defaultdict): The values for each state. V[state] = value.
    """
    V = defaultdict(float)
    N = defaultdict(int)

    for episode in episodes:
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            # Update V and N here according to first visit MC
            reward = episode[t][2]
            G = gamma * G + reward
            current_state = episode[t][0]
            previous_states = [i[0] for i in episode[0:t]]
            if current_state not in previous_states:
                N[current_state]+=1
                V[current_state]+= 1/N[current_state] * (G-V[current_state])
    return V

def learning_targets(
    V: defaultdict, gamma: float, episodes, n: Optional[int] = None
) -> np.ndarray:
    """Compute the learning targets for the given evaluation episodes.

    This generic function computes the learning targets for Monte Carlo (n=None), TD(0) (n=1), or TD(n) (n=n).

    Args:
        V (defaultdict) : A dict of state values
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int or None): The number of steps for the learning targets. Use n=1 for TD(0), n=None for MC.
    """
    targets = []
    if n is None:
        for episode in episodes:
            G = 0.0
            for t in range(len(episode)-1,-1,-1):
                state,action,reward = episode[t][0],episode[t][1],episode[t][2]
                G += gamma * reward
                if state == (3,0):
                    targets.append(G)
        return targets
    elif n == 1:
        for episode in episodes:
            next_state,reward = episode[1][0],episode[0][2]
            target = reward + gamma * V[next_state]
            targets.append(target)
        return targets
    elif n >= 4: 
        for episode in episodes:
            target = 0.0
            n_state = episode[n][0]
            for i in range(n):
                reward = episode[i][2]
                target += np.power(gamma,i) * reward
            target+= np.power(gamma,n) * V[n_state]
            targets.append(target)
        return targets
    
