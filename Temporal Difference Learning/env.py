from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np


def register_env() -> None:
    """Register custom gym environment so that we can use `gym.make()`

    In your main file, call this function before using `gym.make()` to use the Four Rooms environment.
        register_env()
        env = gym.make('WindyGridWorld-v0')

    There are a couple of ways to create Gym environments of the different variants of Windy Grid World.
    1. Create separate classes for each env and register each env separately.
    2. Create one class that has flags for each variant and register each env separately.

        Example:
        (Original)     register(id="WindyGridWorld-v0", entry_point="env:WindyGridWorldEnv")
        (King's moves) register(id="WindyGridWorldKings-v0", entry_point="env:WindyGridWorldEnv", **kwargs)

        The kwargs will be passed to the entry_point class.

    3. Create one class that has flags for each variant and register env once. You can then call gym.make using kwargs.

        Example:
        (Original)     gym.make("WindyGridWorld-v0")
        (King's moves) gym.make("WindyGridWorld-v0", **kwargs)

        The kwargs will be passed to the __init__() function.

    Choose whichever method you like.
    """
    register(id="WindyGridWorld-v0",entry_point = "env:WindyGridWorldEnv")
    register(id = "WindyGridWorldKings-v0", entry_point = "env:WindyGridWorldKingsEnv")
    register(id = "WindyGridWorldKings-v1", entry_point = "env:StochWindyGridWorldKingsEnv")


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (0, -1),
        Action.DOWN: (1, 0),
        Action.RIGHT: (0, 1),
        Action.UP: (-1, 0),
    }
    return mapping[action]


class WindyGridWorldEnv(Env):
    def __init__(self):
        """Windy grid world gym environment
        """
        # Grid dimensions (x, y)
        self.rows = 7
        self.cols = 10

        # Wind
        #define self.wind as either a dict (keys would be states) or multidimensional array (states correspond to indices)
        self.wind = np.zeros((self.rows, self.cols),dtype=int)
        self.wind[:, [3, 4, 5, 8]] = 1
        self.wind[:, [6, 7]] = 2
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

        # Set start_pos and goal_pos
        self.start_pos = (3, 0)
        self.goal_pos = (3, 7)
        self.agent_pos = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """
        done = False
        reward = -1
        dx, dy = actions_to_dxdy(action)
        x  = min(max((self.agent_pos[0] + dx - self.wind[self.agent_pos]),0), self.rows-1)
        y = min(max(self.agent_pos[1] + dy, 0), self.cols-1)
        self.agent_pos = (x, y)
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, -1, True, {}
        return self.agent_pos, reward, done, {}

###########################################################################################
#Gym environment for windy grid world with Kings moves
class KingAction(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    LEFT_UP = 4
    RIGHT_UP = 5
    LEFT_DOWN = 6
    RIGHT_DOWN = 7
    # STAY = 8


def king_actions_to_dxdy(action: KingAction) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (KingAction): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        KingAction.LEFT: (0, -1),
        KingAction.DOWN: (1, 0),
        KingAction.RIGHT: (0, 1),
        KingAction.UP: (-1, 0),
        KingAction.LEFT_UP: (-1, -1),
        KingAction.RIGHT_UP: (-1, 1),
        KingAction.LEFT_DOWN: (1, -1),
        KingAction.RIGHT_DOWN: (1, 1),
        # KingAction.STAY: (0,0),
    }
    return mapping[action]


class WindyGridWorldKingsEnv(Env):
    def __init__(self):
        """Windy grid world gym environment with Kings moves (8 actions)
        """
        # Grid dimensions (x, y)
        self.rows = 7
        self.cols = 10

        # Wind
        #define self.wind as either a dict (keys would be states) or multidimensional array (states correspond to indices)
        self.wind = np.zeros((self.rows, self.cols),dtype=int)
        self.wind[:, [3, 4, 5, 8]] = 1
        self.wind[:, [6, 7]] = 2
        self.action_space = spaces.Discrete(len(KingAction))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

        # Set start_pos and goal_pos
        self.start_pos = (3, 0)
        self.goal_pos = (3, 7)
        self.agent_pos = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """
        done = False
        reward = -1
        dx, dy = king_actions_to_dxdy(action)
        x  = min(max((self.agent_pos[0] + dx - self.wind[self.agent_pos]),0), self.rows-1)
        y = min(max(self.agent_pos[1] + dy, 0), self.cols-1)   
        self.agent_pos = (x, y)
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, -1, True, {}
        return self.agent_pos, reward, done, {}

###########################################################################################
#Gym environment for windy grid world with Kings moves and stochastic wind
class StochWindyGridWorldKingsEnv(Env):
    def __init__(self):
        """Windy grid world gym environment with Kings moves (8 actions)
        """
        # Grid dimensions (x, y)
        self.rows = 7
        self.cols = 10

        # Wind
        #define self.wind as either a dict (keys would be states) or multidimensional array (states correspond to indices)
        self.wind = np.zeros((self.rows, self.cols),dtype=int)
        self.wind[:, [3, 4, 5, 8]] = 1
        self.wind[:, [6, 7]] = 2
        self.action_space = spaces.Discrete(len(KingAction))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

        # Set start_pos and goal_pos
        self.start_pos = (3, 0)
        self.goal_pos = (3, 7)
        self.agent_pos = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """
        done = False
        reward = -1
        dx, dy = king_actions_to_dxdy(action)
        x  = min(max((self.agent_pos[0] + dx - self.wind[self.agent_pos]+np.random.choice([-1,0,1])),0), self.rows-1)
        y = min(max(self.agent_pos[1] + dy, 0), self.cols-1)   
        self.agent_pos = (x, y)
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, -1, True, {}
        return self.agent_pos, reward, done, {}



