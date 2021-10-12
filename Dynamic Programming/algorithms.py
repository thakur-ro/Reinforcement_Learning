from env import Action,Gridworld5x5,JacksCarRental
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class Q5:

    def __init__(self,gamma:float,theta:float) -> None:
        self.gamma = gamma
        self.theta = theta
    
    def iterative_policy_evaluation(self,gamma:float,theta:float,policy:np.ndarray) -> list:
        grid_world_env = Gridworld5x5()
        n = len(grid_world_env.state_space)
        V = np.zeros(n,dtype = float)
        V_updated = V
        while True:
            delta = 0.0
            V_updated = np.zeros(n,dtype = float)
            for i,state in enumerate(grid_world_env.state_space):
                v = V[i]
                for j,action in enumerate(Action):
                    V_updated[i] += policy[i][j]*grid_world_env.expected_return(V,state,action,self.gamma) 
                delta  = max(delta,abs(v-V_updated[i]))
            V = V_updated
            if (delta < self.theta):
                return V

    def value_iteration(self,gamma:float,theta:float) -> (np.ndarray,defaultdict(list)):
        grid_world_env = Gridworld5x5()
        n = len(grid_world_env.state_space)
        V = np.zeros(n,dtype = float)
        policy = defaultdict(list)
        while True:
            delta = 0.0
            for i,state in enumerate(grid_world_env.state_space):
                v = V[i]

                action_value = {'LEFT':grid_world_env.expected_return(V,state,Action.LEFT,self.gamma),
                'DOWN':grid_world_env.expected_return(V,state,Action.DOWN,self.gamma),
                'RIGHT':grid_world_env.expected_return(V,state,Action.RIGHT,self.gamma),
                'UP':grid_world_env.expected_return(V,state,Action.UP,self.gamma)}
                V[i] = max(action_value.values())

                delta = max(delta,abs(v-V[i]))
            if (delta < self.theta):
                break
        
        for i,state in enumerate(grid_world_env.state_space):
            action_val = {'LEFT':0,'DOWN':0,'RIGHT':0,'UP':0}
            for action in Action:
                action_val[action.name] = round(grid_world_env.expected_return(V,state,action,self.gamma),2)
            max_value = max(action_val.values())
            optimal_action = [k for k,v in action_val.items() if v == max_value]
            for action in optimal_action:
                policy[state].append(action)

        return V,policy

        
    def policy_iteration(self,gamma:float,theta:float,policy:np.ndarray):
        grid_world_env = Gridworld5x5()
        n = len(grid_world_env.state_space)
        #Policy Improvement
        while True:
            policy_stable = True
            V = self.iterative_policy_evaluation(gamma,theta,policy)
            for i,state in enumerate(grid_world_env.state_space):
                action_val = {'LEFT':0,'DOWN':0,'RIGHT':0,'UP':0}
                for action in Action:
                    action_val[action.name]  = round(grid_world_env.expected_return(V,state,action,self.gamma),2)
                max_value = max(action_val.values())
                optimal_action = np.array([v if v == max_value else 0.0 for k,v in action_val.items()])
                probabilities = 1.0/np.count_nonzero(optimal_action)
                new_action = np.array([0.0,0.0,0.0,0.0])
                for index,action in enumerate(optimal_action):
                    if action==0.0:
                        continue
                    else:
                        new_action[index] = probabilities
                        
                if not(np.array_equal(policy[i],new_action)):
                    policy_stable = False
                    policy[i] = new_action
            if policy_stable == True:
                break
        
        return V,policy

class Q6:
  
  def __init__(self,gamma:float,theta:float,modified:bool) -> None:
    self.gamma = gamma 
    self.theta = theta
    self.modified = modified
  
  def argmax(self,action_val):
    max_val = np.max(action_val)
    nonzero = np.count_nonzero(action_val == max_val)
    probabilities = 1.0/nonzero
    new_actions = np.zeros(11)
    for i,val in enumerate(action_val):
      if val == max_val:
        new_actions[i] = probabilities
    return new_actions


  def iterative_policy_evaluation(self,gamma:float,theta:float,policy:np.ndarray) -> list:
      jack_env = JacksCarRental(self.modified)
      n = len(jack_env.state_space)
      V = np.zeros((21,21),dtype=float)
      jack_env.precompute_transitions()
      actions = jack_env.action_space
      while True:
          delta = 0.0
          V_updated = np.zeros((21,21),dtype = float)
          for i,state in enumerate(jack_env.state_space):
              action = policy[state]
              v = V[state]
              V_updated[state] = jack_env.expected_return(V,state,action,self.gamma) 
              delta  = max(delta,abs(v-V_updated[state]))
          V = V_updated
          if (delta < self.theta):
              return V

  def policy_iteration(self,gamma:float,theta:float,policy:np.ndarray):
      jack_env = JacksCarRental(self.modified)
      jack_env.precompute_transitions() 
      actions = jack_env.action_space
      n = len(jack_env.state_space)
      #Policy Improvement
      while True:
          policy_stable = True
          V = self.iterative_policy_evaluation(gamma,theta,policy)
          for i,state in enumerate(jack_env.state_space):
              action_val = np.zeros(len(actions))
              for j,action in enumerate(actions):
                action_val[j]  = jack_env.expected_return(V,state,action,self.gamma)
              new_action = actions[np.argmax(action_val)]                
              current_action = policy[state]
              if (current_action!=new_action):
                policy_stable = False
              policy[state] = new_action
          if policy_stable:
              break
      return V,policy