from abc import ABCMeta, abstractmethod
import random
import numpy as np
from gym.spaces import Dict
from gym.spaces import MultiBinary
import gym
import copy
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager

MCTS_FACTOR = 500 # send to agent init
ROLLOUT_FACTOR = 60 # send to agent init
C_UCB_PARAM = np.sqrt(2) # send to agent init

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

    def sample_action(self, env=None):
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

    def sample_action(self, env):
        s = self.__GetActionWithMCTS(env)
        action = {}
        selected_actions = random.sample(list(s), self.num_actions)
        for sample in selected_actions:
            if isinstance(self.action_space[sample], gym.spaces.Box):
                action[sample] = s[sample][0].item()
            elif isinstance(self.action_space[sample], gym.spaces.Discrete):
                action[sample] = s[sample]
        return action
    
    def __GetActionWithMCTS(self, env):
        root = MonteCarloTreeSearchNode(env = env,
                                        action_space = self.action_space)
        return root.best_action()

class MonteCarloTreeSearchNode():
    def __init__(self, env, action_space, parent=None, parent_action=None):
        self.__myEnv = env
        self.__parent = parent
        self.parent_action = parent_action
        self.__children = []
        self.number_of_visits = 0
        self.score = 0

        # TODO - should we use Dict?
        self.__untried_actions = [0,1]

        self.__action_space = action_space
        return
    
    def expand(self):
        #TODO - how can we remove the action we already tried?
        action = self.__untried_actions.pop()

        EnvInfo = ExampleManager.GetEnvInfo("Traffic")
        new_env = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), 
                            instance=EnvInfo.get_instance(0))
        new_env.sampler.subs = copy.deepcopy(self.__myEnv.sampler.subs)

        next_state, reward, done, info = new_env.step({'advance___i0': action})
        child_node = MonteCarloTreeSearchNode(env = new_env,
                                            action_space = self.__action_space,  
                                            parent=self, 
                                            parent_action={'advance___i0': action})
        child_node.backpropagate(reward)
        self.__children.append(child_node)
        return child_node
    
    def rollout(self):
        EnvInfo = ExampleManager.GetEnvInfo("Traffic")
        current_myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), 
                            instance=EnvInfo.get_instance(0))
        current_myEnv.sampler.subs = copy.deepcopy(self.__myEnv.sampler.subs)

        final_reward = 0
        for i in range(ROLLOUT_FACTOR):
            action = self.__rollout_policy(self.__action_space)
            current_rollout_state, reward, done, info = current_myEnv.step(action)
            final_reward += reward
        return final_reward
    
    def backpropagate(self, result):
        self.number_of_visits += 1
        self.score += result
        if self.__parent:
            self.__parent.backpropagate(result)

    def is_fully_expanded(self):

        #TODO - is this the right way to check if the node is fully expanded?
        return len(self.__untried_actions) == 0
    
    def best_child(self, c_ucb_param = C_UCB_PARAM):
        choices_weights = []
        for child in self.__children:
            if child.number_of_visits == 0:
                return child
            else:
                choices_weights.append(child.__UCB1(c_ucb_param)) # TODO - why __UCB1 is not recognized?
        return self.__children[np.argmax(choices_weights)]
    
    def __UCB1(self, c_ucb_param):
        return (self.score / self.number_of_visits) + c_ucb_param * np.sqrt((2 * np.log(self.__parent.number_of_visits) / self.number_of_visits))

    def __rollout_policy(self, possible_moves):
        return possible_moves.sample() # random move
    
    def __tree_policy(self):
        current_node = self
        while True:
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
    
    def best_action(self):
        for i in range(MCTS_FACTOR):    
            v = self.__tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        return self.best_child().parent_action