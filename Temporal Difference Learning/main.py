import itertools
import matplotlib.pyplot as plt
import gym
from algorithms import sarsa,q_learning,exp_sarsa,nstep_sarsa,on_policy_mc_control_epsilon_soft,td_prediction,generate_episode,mc_prediction,learning_targets
import numpy as np
from env import register_env
import tqdm
from itertools import product

def plot_environment(sarsa,ql,exp_sarsa,nsarsa,mc,env_name):
    mean_sarsa = np.mean(sarsa,axis=0)
    mean_ql = np.mean(ql,axis=0)
    mean_exp = np.mean(exp_sarsa,axis=0)
    mean_nsarsa = np.mean(nsarsa,axis=0)
    mean_mc = np.mean(mc,axis=0)

    #find standard deviations
    std_sarsa = np.std(sarsa,axis=0)
    std_ql = np.std(ql,axis=0)
    std_exp = np.std(exp_sarsa,axis=0)
    std_nsarsa = np.std(nsarsa,axis=0)
    std_mc = np.std(mc,axis=0)

    #find confidence interval of mean_sarsa
    conf_sarsa = 1.96 * std_sarsa / np.sqrt(10)
    conf_ql = 1.96 * std_ql / np.sqrt(10)
    conf_exp = 1.96 * std_exp / np.sqrt(10)
    conf_nsarsa = 1.96 * std_nsarsa / np.sqrt(10)
    conf_mc = 1.96 * std_mc / np.sqrt(10)

    plt.xlabel('Timesteps')
    plt.ylabel('Episodes')
    plt.plot(mean_sarsa, label='Average SARSA Episodes Completed')
    plt.plot(mean_ql, label='Average Q-learning Episodes Completed')
    plt.plot(mean_exp, label='Average Expected SARSA Episodes Completed')
    plt.plot(mean_nsarsa, label='Average N-step SARSA Episodes Completed')
    plt.plot(mean_mc, label='Average e-soft Monte Carlo Episodes Completed')
    plt.fill_between(np.arange(0,8000), mean_sarsa - conf_sarsa, mean_sarsa + conf_sarsa, alpha=0.2, color='b')
    plt.fill_between(np.arange(0,8000), mean_ql - conf_ql, mean_ql + conf_ql, alpha=0.2, color='r')
    plt.fill_between(np.arange(0,8000), mean_exp - conf_exp, mean_exp + conf_exp, alpha=0.2, color='g')
    plt.fill_between(np.arange(0,8000), mean_nsarsa - conf_nsarsa, mean_nsarsa + conf_nsarsa, alpha=0.2, color='y')
    plt.fill_between(np.arange(0,8000), mean_mc - conf_mc, mean_mc + conf_mc, alpha=0.2, color='k')
    plt.xticks(np.arange(0, 8001, 1000))
    plt.title(env_name+" Environment")
    plt.legend()
    plt.savefig('outputs/'+env_name+'.png')
    plt.close()

def windy_grid_world():
    register_env()
    env = gym.make('WindyGridWorld-v0')
    env.seed(0)
    episodes_completed_sarsa=[]
    episodes_completed_ql=[]
    episodes_completed_exp=[]
    episodes_completed_nsarsa=[]
    episodes_completed_mc=[]
    trials = 10
    for trial in tqdm.trange(trials, desc='Trials', leave=False):
        episodes_completed_sarsa.append(sarsa(env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        episodes_completed_ql.append(q_learning(env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        episodes_completed_exp.append(exp_sarsa(env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        episodes_completed_nsarsa.append(nstep_sarsa(env, num_steps=8000, n=4,gamma=1, epsilon=0.1, step_size=0.5)[1])
        episodes_completed_mc.append(on_policy_mc_control_epsilon_soft(env, num_steps=8000, gamma=1, epsilon=0.1)[1])
    plot_environment(episodes_completed_sarsa,episodes_completed_ql,episodes_completed_exp,episodes_completed_nsarsa,\
        episodes_completed_mc,'Windy Grid World')
        
def kings_grid_world():
    register_env()
    env = gym.make('WindyGridWorldKings-v0')
    env.seed(0)
    trials = 10
    episodes_completed_sarsa=[]
    episodes_completed_ql=[]
    episodes_completed_exp=[]
    episodes_completed_nsarsa=[]
    episodes_completed_mc=[]
    for trial in tqdm.trange(trials, desc='Trials', leave=False):
        episodes_completed_sarsa.append(sarsa(env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        episodes_completed_ql.append(q_learning(env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        episodes_completed_exp.append(exp_sarsa(env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        episodes_completed_nsarsa.append(nstep_sarsa(env, num_steps=8000, n=4,gamma=1, epsilon=0.1, step_size=0.5)[1])
        episodes_completed_mc.append(on_policy_mc_control_epsilon_soft(env, num_steps=8000, gamma=1, epsilon=0.1)[1])
    plot_environment(episodes_completed_sarsa,episodes_completed_ql,episodes_completed_exp,episodes_completed_nsarsa,\
        episodes_completed_mc,'Windy Grid World Kings Move(with 8 steps)')



def stoch_kings_grid_world():
    register_env()
    env = gym.make('WindyGridWorldKings-v1')
    env.seed(0)
    trials = 10
    episodes_completed_sarsa=[]
    episodes_completed_ql=[]
    episodes_completed_exp=[]
    episodes_completed_nsarsa=[]
    episodes_completed_mc=[]
    for trial in tqdm.trange(trials, desc='Trials', leave=False):
        episodes_completed_sarsa.append(sarsa(env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        episodes_completed_ql.append(q_learning(env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        episodes_completed_exp.append(exp_sarsa(env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        episodes_completed_nsarsa.append(nstep_sarsa(env, num_steps=8000, n=4,gamma=1, epsilon=0.1, step_size=0.5)[1])
        episodes_completed_mc.append(on_policy_mc_control_epsilon_soft(env, num_steps=8000, gamma=1, epsilon=0.1)[1])
    plot_environment(episodes_completed_sarsa,episodes_completed_ql,episodes_completed_exp,episodes_completed_nsarsa,\
        episodes_completed_mc,'Stochastic Windy Grid World Kings Move(with 8 steps)')

def Q5():
    N = [1,10,50]
    n_t = [1,4]
    gamma,step_size,epsilon,num_steps = 1,0.5,0.1,8000
    register_env()
    env = gym.make('WindyGridWorld-v0')
    V_td = []
    V_mc = []
    Q,_ = q_learning(env, num_steps=num_steps, gamma=gamma, epsilon=epsilon, step_size=step_size)
    for n in N:
        episodes = []
        for i in range(n):
            episodes.append(generate_episode(env,Q,epsilon,num_steps)[0])
        V_1 = td_prediction(env,gamma,step_size,episodes,n=1)
        V_4 = td_prediction(env,gamma,step_size,episodes,n=4)
        V_mc_1 = mc_prediction(episodes,gamma)
        V_td.append(V_1)
        V_td.append(V_4)
        V_mc.append(V_mc_1)

    V_1_1 = V_td[0]
    V_1_4 = V_td[1]
    V_10_1 = V_td[2]
    V_10_4 = V_td[3]
    V_50_1 = V_td[4]
    V_50_4 = V_td[5]
    V_1_mc = V_mc[0]
    V_10_mc = V_mc[1]
    V_50_mc = V_mc[2]

    #generate evaluation episodes 
    episodes_eval = []
    for i in range(100):
        episodes_eval.append(generate_episode(env,Q,epsilon,num_steps)[0])
    target_mc_1 = learning_targets(V_1_mc,gamma,episodes_eval,n=None)
    target_mc_10 = learning_targets(V_10_mc,gamma,episodes_eval,n=None)
    target_mc_50 = learning_targets(V_50_mc,gamma,episodes_eval,n=None)

    target_td_1 = learning_targets(V_1_1,gamma,episodes_eval,n=1)
    target_td_10 = learning_targets(V_10_1,gamma,episodes_eval,n=1)
    target_td_50 = learning_targets(V_50_1,gamma,episodes_eval,n=1)

    target_td_14 = learning_targets(V_1_4,gamma,episodes_eval,n=4)
    target_td_104 = learning_targets(V_10_4,gamma,episodes_eval,n=4)
    target_td_504 = learning_targets(V_50_4,gamma,episodes_eval,n=4)

    #plot learning targets as histogram for TD(0)
    plt.figure(figsize=(10,10))
    plt.hist(target_td_1,label='Episodes=1 TD(0)')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    ax = plt.gca()
    ax.invert_xaxis()
    plt.savefig('outputs/Q5_1_td.png')
    plt.close()
    plt.hist(target_td_10,label='Episodes=10 TD(0)')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    ax = plt.gca()
    ax.invert_xaxis()    
    plt.savefig('outputs/Q5_10_td.png')
    plt.close()
    plt.hist(target_td_50,label='Episodes=50 TD(0)')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    ax = plt.gca()
    ax.invert_xaxis()
    plt.savefig('outputs/Q5_50_td.png')
    plt.close()

    #plot learning targets as histogram for TD(4)
    plt.figure(figsize=(10,10))
    plt.hist(target_td_14,label='Episodes=1 TD(4)')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    ax = plt.gca()
    ax.invert_xaxis()
    plt.savefig('outputs/Q5_1_td_4.png')
    plt.close()
    plt.hist(target_td_104,label='Episodes=10 TD(4)')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    ax = plt.gca()
    ax.invert_xaxis()
    plt.savefig('outputs/Q5_10_td_4.png')
    plt.close()
    plt.hist(target_td_504,label='Episodes=50 TD(4)')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    ax = plt.gca()
    ax.invert_xaxis()
    plt.savefig('outputs/Q5_50_td_4.png')
    plt.close()

    #plot learning targets as histogram for MC
    plt.figure(figsize=(10,10))
    plt.hist(target_mc_1,label='Episodes=1 MC')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    ax = plt.gca()
    ax.invert_xaxis()
    plt.savefig('outputs/Q5_1_mc.png')
    plt.close()
    plt.hist(target_mc_10,label='Episodes=10 MC')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    ax = plt.gca()
    ax.invert_xaxis()
    plt.savefig('outputs/Q5_10_mc.png')
    plt.close()
    plt.hist(target_mc_50,label='Episodes=50 MC')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    ax = plt.gca()
    ax.invert_xaxis()
    plt.savefig('outputs/Q5_50_mc.png')
    plt.close()
def main():
    windy_grid_world()
    kings_grid_world()
    stoch_kings_grid_world()
    Q5()
    
if __name__ == "__main__":
    main()