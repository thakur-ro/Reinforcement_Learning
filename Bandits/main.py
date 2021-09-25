from env import BanditEnv
from tqdm import trange
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from agent import EpsilonGreedy,UCB
import numpy as np

def q4(k: int, num_samples: int):
    """Q4

    Structure:
        1. Create multi-armed bandit env
        2. Pull each arm `num_samples` times and record the rewards
        3. Plot the rewards (e.g. violinplot, stripplot)

    Args:
        k (int): Number of arms in bandit environment
        num_samples (int): number of samples to take for each arm
    """
    """10 arm test bed"""
    rewards = defaultdict(list)
    env = BanditEnv(k=k)
    env.reset()
    for i in range(k):
        for j in range(num_samples):
            rewards[i].append(env.step(i))
    rewards_df = pd.DataFrame(rewards)

    fig, axes = plt.subplots()
    axes.violinplot(dataset = rewards_df,showmeans=True,showextrema=False)
    axes.set_title('10-Armed Testbed')
    axes.yaxis.grid(True)
    axes.set_xlabel('Action')
    axes.set_ylabel('Reward Distribution')
    axes.set_xticks([i+1 for i in range(10)])
    plt.show()

    pass


def q6(k: int, trials: int, steps: int):
    """Q6

    Implement epsilon greedy bandit agents with an initial estimate of 0

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    #initialize env and agents
    env = BanditEnv(k=k)
    env.reset()
    agents = [EpsilonGreedy(k=k,init=0,epsilon=eps) for eps in [0.01,0.1,0]] #0.01,0.1,0
    
    rewards = np.zeros((trials,len(agents),steps))
    optimal = np.zeros((trials,len(agents),steps))
    opt_reward = np.zeros((trials,steps))
    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        for position, agent in enumerate(agents):
            agent.reset()
            for step in range(steps):
                action = agent.choose_action()
                if action == env.optimal_action: 
                    optimal[t][position][step] = 1
                step_reward = env.step(action)
                rewards[t][position][step] = step_reward
                agent.update(action,step_reward)
                opt_reward[t][step] = env.optimal_reward

    sample_average = np.mean(rewards,axis=0)
    optimal_average = np.mean(optimal, axis=0)
    opt_average = np.mean(opt_reward,axis=0)
    colors = ['b','orange','g']    
    lower = np.mean(rewards,axis=0) - np.divide(np.std(rewards,axis=0),np.sqrt(trials))
    upper = np.mean(rewards,axis=0) + np.divide(np.std(rewards,axis=0),np.sqrt(trials))
    for pos,i in enumerate(sample_average):
        plt.plot(range(steps),i)
        plt.fill_between(range(steps),lower[pos],upper[pos],color=colors[pos],alpha=0.3)

    opt_std_err = 1.96 * np.divide(np.std(opt_reward,axis=0),np.sqrt(trials))   
    plt.plot(range(steps),opt_average,color='k')
    plt.fill_between(range(steps),opt_average - opt_std_err,opt_average + opt_std_err, color=colors[pos],alpha=0.3)
    plt.title("Epsilon Greedy")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend(['epsilon = 0.01','epsilon = 0.1','epsilon = 0','Best Possible Average Performance'])
    plt.show()

"""Fig 2.2"""
    for i in optimal_average:
        plt.plot(range(steps),i)
    plt.title("Epsilon Greedy-Optimal Actions Taken")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Actions")
    plt.legend(['epsilon = 0.01','epsilon = 0.1','epsilon = 0'])
    plt.show()




def q7(k: int, trials: int, steps: int):
    """Q7

    Compare epsilon greedy bandit agents and UCB agents

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # initialize env and agents
    env = BanditEnv(k=k)
    epsilon = [0,0,0.1,0.1]
    init = [0,5,0,5]
    agents=[]
    agents = [EpsilonGreedy(k=k,init=i,epsilon=e,step_size=0.1) for e,i in zip(epsilon,init)]
    agents_ucb = agents.append(UCB(k=k,init=0,c=2,step_size=0.1))

    rewards = np.zeros((trials,len(agents),steps))
    optimal = np.zeros((trials,len(agents),steps)) #For optimal Actions
    opt_reward = np.zeros((trials,steps))  #For optimal rewards max(rewards)
    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        for position,agent in enumerate(agents):
            agent.reset()
            for step in range(steps):
                action = agent.choose_action()
                #check if action is optimal, and update optimal array
                if action == env.optimal_action: 
                    optimal[t][position][step] = 1
                #Take step into environment (Pull Arm)
                step_reward = env.step(action)
                #Record reward
                rewards[t][position][step] = step_reward
                #Update Agent
                agent.update(action,step_reward)
                #Record Optimal Rewards
                opt_reward[t][step] = env.optimal_reward
    sample_average = np.mean(rewards,axis=0)
    optimal_average = np.mean(optimal, axis=0)
    opt_average = np.mean(opt_reward,axis=0)

    #Plot optimal Action
    for pos,i in enumerate(optimal_average):
        plt.plot(range(steps),i)
    plt.title("Epsilon Greedy-Optimal Actions Taken")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Actions")
    plt.legend(['Q = 0,epsilon = 0','Q = 5, epsilon = 0','Q = 0,epsilon = 0.1','Q = 5, epsilon = 0.1','C=2'])
    plt.show()


    #plot rewards
    lower = np.mean(rewards,axis=0) - np.divide(np.std(rewards,axis=0),np.sqrt(trials))
    upper = np.mean(rewards,axis=0) + np.divide(np.std(rewards,axis=0),np.sqrt(trials))
    for pos,i in enumerate(sample_average):
        plt.plot(range(steps),i)
        plt.fill_between(range(steps),lower[pos],upper[pos],alpha=0.3)

    opt_std_err = 1.96 * np.divide(np.std(opt_reward,axis=0),np.sqrt(trials))   
    plt.plot(range(steps),opt_average,color='k')
    plt.fill_between(range(steps),opt_average - opt_std_err,opt_average + opt_std_err,alpha=0.3)
    plt.title("Average Reward Comparison")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend(['Q = 0,epsilon = 0','Q = 5, epsilon = 0','Q = 0,epsilon = 0.1','Q = 5, epsilon = 0.1','C=2',
    'Best Possible Average Performance'])
    plt.show()


def main():
    # run code for all questions
    q4(k=10,num_samples = 10000)
    q6(k=10,trials= 2000,steps=1000)
    q7(k=10,trials= 2000,steps=1000)
    pass


if __name__ == "__main__":
    main()
