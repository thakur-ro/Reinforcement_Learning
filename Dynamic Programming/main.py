from env import JacksCarRental
from algorithms import Q5,Q6
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
from pprint import pprint
from mpl_toolkits.mplot3d import Axes3D

def question_5(gamma:float,theta:float) -> None:
    q5 = Q5(gamma,theta)
    random_policy = 0.25 * np.ones((25,4),dtype = float)
    value_function = q5.iterative_policy_evaluation(gamma,theta,random_policy) ##Iterative Policy Evaluation
    print(colored("\n\nValue Function for iterative policy evaluation\n",'red') +colored('{}'.format(np.round(value_function,1)),'green'))

    optimal_value,policy = q5.value_iteration(gamma,theta) #Value Iteration algorithm
    print(colored("\nOptimal Value function for Value Iteration \n",'red') + colored("{}".format(np.round(optimal_value,1)),'green'))
    pprint("Optimal policy for Value Iteration \n {}".format(policy))

    optimal_value,optimal_policy = q5.policy_iteration(gamma,theta,random_policy)
    print(colored("\nOptimal Value function for Policy Iteration \n",'red') + colored("{}".format(np.round(optimal_value,1)),'green'))
    pprint("Optimal policy function for Policy Iteration \n {}".format(optimal_policy))

def question_6(gamma:float,theta:float,modified:bool) -> None:
    q6 = Q6(gamma,theta,modified)
    initial_policy = np.zeros((21,21),dtype=int)

    optimal_value,optimal_policy = q6.policy_iteration(gamma,theta,initial_policy)
    
    with open('/content/drive/MyDrive/ex3/jack_policy_1.npy', 'wb') as f:
      np.save(f, optimal_policy)
    with open('/content/drive/MyDrive/ex3/jack_value_1.npy', 'wb') as f:
      np.save(f, optimal_value)


    with open('/content/drive/MyDrive/ex3/jack_policy_1.npy', 'rb') as f:
      policy = np.load(f)
    with open('/content/drive/MyDrive/ex3/jack_value_1.npy', 'rb') as f:
      value = np.load(f)
    plt.imshow(policy,origin='lower')
    for (j,i),label in np.ndenumerate(policy):
        plt.text(i,j,label,ha='center',va='center',color='white')
    plt.colorbar()
    plt.plot()
    plt.savefig("/content/drive/MyDrive/ex3/policy_original.png")

    (x, y) = np.meshgrid(np.arange(value.shape[0]), np.arange(value.shape[1]))
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1,projection='3d')
    surf = ax.plot_surface(x, y, value,cmap=plt.cm.coolwarm)
    fig.colorbar(surf)
    fig.show()
    plt.savefig("/content/drive/MyDrive/ex3/value_original.png")


def main():
    question_5(gamma=0.9,theta = 0.001) #Grid World Environment
    question_6(gamma=0.9,theta=0.001,modified = False) #Jack's Car rental Environment

if __name__ == "__main__":
    main()