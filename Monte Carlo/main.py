import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import gym
from tqdm.std import trange
from algorithms import on_policy_mc_evaluation,on_policy_mc_control_es,on_policy_mc_control_epsilon_soft
import policy
import numpy as np
from collections import defaultdict
from env import register_env
import tqdm

def convert_to_array(V:defaultdict,flag_ace:bool):
    V_arr = np.zeros([10, 10])   
    #Normalizing values to 0-9 for plotting
    for i in range(12, 22):
        for j in range(1, 11):
            V_arr[i-12,j-1] = V[(i,j,flag_ace)]
    return V_arr

def plot_wireframe(V:defaultdict,ax:axes3d,flag_ace:bool):
    V_arr = convert_to_array(V,flag_ace)
    dealer_showing = list(range(1,11))
    player_sum  = list(range(12,22))
    X,Y = np.meshgrid(dealer_showing,player_sum)
    ax.plot_wireframe(X,Y,V_arr)

def fig5_1():
    env = gym.make('Blackjack-v1')
    V_10k = on_policy_mc_evaluation(env,policy = policy.default_blackjack_policy, num_episodes=10000,gamma=1)
    V_500k = on_policy_mc_evaluation(env,policy = policy.default_blackjack_policy, num_episodes=500000,gamma=1)
    fig = plt.figure(figsize=(20,8))
    ax_10k_no_ace = fig.add_subplot(2,2,3, projection='3d')
    ax_10k_has_ace = fig.add_subplot(2,2,1, projection='3d', title='After 10,000 episodes')
    ax_500k_no_ace = fig.add_subplot(2,2,4, projection='3d')
    ax_500k_has_ace = fig.add_subplot(2,2,2, projection='3d', title='After 500,000 episodes')

    ax_500k_no_ace.set_xlabel('Dealer Showing')
    ax_10k_no_ace.set_ylabel('Player Sum')
    fig.text(0.1, 0.75, 'Usable\n  Ace', fontsize=12)
    fig.text(0.1, 0.25, '   No\nUsable\n  Ace', fontsize=12)

    plot_wireframe(V_10k,ax_10k_no_ace,flag_ace=False)
    plot_wireframe(V_10k,ax_10k_has_ace,flag_ace=True)
    plot_wireframe(V_500k,ax_500k_no_ace,flag_ace=False)
    plot_wireframe(V_500k,ax_500k_has_ace,flag_ace=True)

    plt.tight_layout()

    plt.savefig('fig5_1.png')
    plt.show()
    plt.close()

def get_arrays(Q:defaultdict,pi:callable):
    pi_no_ace = np.zeros([10,10])
    pi_has_ace = np.zeros([10,10])
    v_no_ace = np.zeros([10,10])
    v_has_ace = np.zeros([10,10])
    
    for i in range(12,22):
        for j in range(1,11):
            pi_no_ace[i-12,j-1] = pi((i,j,False))
            pi_has_ace[i-12,j-1] = pi((i,j,True))
            v_no_ace[i-12,j-1] = Q[(i,j,False)][pi((i,j,False))]
            v_has_ace[i-12,j-1] = Q[(i,j,True)][pi((i,j,True))]
    
    return pi_no_ace,pi_has_ace,v_no_ace,v_has_ace

def plot5_2_2d(ax:plt.axes,pi_arr:np.array):
    ax.imshow(pi_arr,origin='lower',vmin=-1,vmax=1,cmap=plt.cm.coolwarm,alpha=0.3,extent=[0.5,10.5,11.5,21.5],interpolation='none')
    ax.set_xticks(np.arange(1,11, 1))
    ax.set_yticks(np.arange(12,22, 1))

def plot5_2_3d(ax:axes3d,v_arr:np.array):
    dealer_showing = list(range(1,11))
    player_sum = list(range(12,22))
    x,y = np.meshgrid(dealer_showing,player_sum)
    ax.plot_wireframe(x,y,v_arr)

def plot5_2(Q:defaultdict,pi:callable):
    pi_no_ace,pi_has_ace,v_no_ace,v_has_ace = get_arrays(Q,pi)

    fig = plt.figure(figsize=(10,10))
    ax_pi_no_ace = fig.add_subplot(2,2,3)
    ax_pi_has_ace = fig.add_subplot(2,2,1, title='Optimal Policy')
    v_5mil_no_ace = fig.add_subplot(2,2,4, projection='3d')
    v_5mil_has_ace = fig.add_subplot(2,2,2, projection='3d', title='Optimal Value')
    ax_pi_has_ace.text(8, 20, 'Stick')
    ax_pi_has_ace.text(8, 13, 'Hit')

    ax_pi_no_ace.set_xlabel('Dealer Showing')
    ax_pi_no_ace.set_ylabel('Player Sum')
    v_5mil_no_ace.set_xlabel('Dealer Showing')
    v_5mil_no_ace.set_ylabel('Player Sum')

    fig.text(0.1, 0.75, 'Usable\n  Ace', fontsize=12)
    fig.text(0.1, 0.25, '   No\nUsable\n  Ace', fontsize=12)


    plot5_2_2d(ax_pi_has_ace,pi_has_ace)
    plot5_2_2d(ax_pi_no_ace,pi_no_ace)
    plot5_2_3d(v_5mil_has_ace,v_has_ace)
    plot5_2_3d(v_5mil_no_ace,v_no_ace)

    plt.savefig('fig5_2.png')
    plt.show()
    plt.close()


def fig5_2():
    env = gym.make('Blackjack-v1')
    Q,pi = on_policy_mc_control_es(env,num_episodes = 3000000,gamma=1)
    plot5_2(Q,pi)

def q4_b(trials: int, episodes: int):
    register_env()
    env = gym.make('FourRooms-v0')
    eps_returns=[]
    for epsilon in [0.1,0.01,0]:
        returns = []
        env.reset()
        for t in trange(trials,desc="trials"):
            env.reset()        
            returns.append(on_policy_mc_control_epsilon_soft(env,num_episodes=episodes,gamma=0.99,epsilon=epsilon))
        eps_returns.append(returns)
    eps_returns = np.array(eps_returns)
    episode_average = np.mean(eps_returns,axis=1)
    print(episode_average.shape)
    e = [0.1,0.01,0]
    colors = ['b','orange','green']    

    lower = np.mean(eps_returns,axis=1) - np.divide(np.std(eps_returns,axis=1),np.sqrt(trials))
    upper = np.mean(eps_returns,axis=1) + np.divide(np.std(eps_returns,axis=1),np.sqrt(trials))
    for pos,val in enumerate(episode_average):
        plt.plot(range(episodes),val,label ='epsilon:'+str(e[pos]),color = colors[pos])
        plt.fill_between(range(episodes),lower[pos],upper[pos],color=colors[pos],alpha=0.3)
    optimal = 0.8179 * np.ones((episodes,))
    plt.plot(range(episodes),optimal,label='optimal bound',color ='red')

    plt.legend()
    plt.xlabel("Number of Episodes")        
    plt.ylabel("Discounted Average Return")
    plt.savefig('q4_b.png')
    plt.show()



def main():
    fig5_1()
    fig5_2()
    q4_b(trials=10,episodes=10000)

if __name__ == "__main__":
    main()