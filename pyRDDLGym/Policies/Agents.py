from abc import ABCMeta, abstractmethod
import random
import numpy as np
from collections import defaultdict
import gym
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager

MCTS_FACTOR = 40
ROLLOUT_FACTOR = 10
C_UCB_PARAM = np.sqrt(2)

class BaseAgent(metaclass=ABCMeta):

    @abstractmethod
    def sample_action(self, state, env):
        pass


class RandomAgent(BaseAgent):
    def __init__(self, action_space, num_actions=1, seed=None):
        self.action_space = action_space
        self.num_actions = num_actions
        if seed is not None:
            self.action_space.seed(seed)

    def sample_action(self, state=None, env=None):
        s = self.action_space.sample()
        action = {}
        selected_actions = random.sample(list(s), self.num_actions)
        for sample in selected_actions:
            if isinstance(self.action_space[sample], gym.spaces.Box):
                action[sample] = s[sample][0].item()
            elif isinstance(self.action_space[sample], gym.spaces.Discrete):
                action[sample] = s[sample]
                # if str(self.action_space[sample]) == 'Discrete(2)':
                #     action[sample] = bool(s[sample])
                # else:
                #     action[sample] = s[sample]
        return action


class NoOpAgent(BaseAgent):
    def __init__(self, action_space, num_actions=0):
        self.action_space = action_space
        self.num_actions = num_actions

    def sample_action(self, state=None):
        action = {}
        return action


class MCTSAgent(BaseAgent):
    def __init__(self, action_space, num_actions=1, seed=None):
        self.action_space = action_space
        self.num_actions = num_actions
        if seed is not None:
            self.action_space.seed(seed)

    def sample_action(self, state, env):
        s = self.action_space.sample()
        action = {}
        selected_actions = self.__GetActionWithMCTS(state, env)
        #NOTE - what we need the for loop for?
        for sample in selected_actions:
            if isinstance(self.action_space[sample], gym.spaces.Box):
                action[sample] = s[sample][0].item()
            elif isinstance(self.action_space[sample], gym.spaces.Discrete):
                action[sample] = s[sample]
        return action
    
    def __GetActionWithMCTS(self, initial_state, env):
        root = MonteCarloTreeSearchNode(env = env,
                                        state = initial_state,
                                        action_space = self.action_space)
        return root.best_action()

class MonteCarloTreeSearchNode():
    def __init__(self, env, state, action_space, score = 0, parent=None, parent_action=None):
        self._myEnv = env
        self._state = state
        self._parent = parent
        self._parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._score = score
        self._untried_actions = action_space
        self._action_space = action_space
        return
    
    def expand(self):
        action = self._untried_actions.pop_random_action() #TODO - find a way to pop a random item from the ordered dict of spaces
        new_env = self._myEnv
        next_state, reward, done, info = new_env.step(action)
        child_node = MonteCarloTreeSearchNode(env = new_env,
                                              state = next_state,
                                              action_space = self._action_space, 
                                              score = reward, 
                                              is_terminal_node = done, 
                                              parent=self, 
                                              parent_action=action)
        self.children.append(child_node)
        return child_node
    
    def rollout(self):
        current_myEnv = self._myEnv
        final_reward = 0
        for i in range(ROLLOUT_FACTOR):
            action = self.rollout_policy(self._action_space)
            current_rollout_state, reward, done, info = current_myEnv.step(action)
            final_reward += reward
        return final_reward
    
    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._score += result
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    def best_child(self, c_ucb_param = C_UCB_PARAM): # add act when number of visits is 0
        choices_weights = [(c._score / c._number_of_visits) + c_ucb_param * np.sqrt((2 * np.log(self._number_of_visits) / c._number_of_visits)) for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]
    
    def _tree_policy(self):
        current_node = self
        while True:
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
    
    def best_action(self):
        for i in range(MCTS_FACTOR):    
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        return self.best_child().parent_action