from abc import ABCMeta, abstractmethod
import random
import numpy as np
from collections import defaultdict
import gym

TIME_TO_MCTS = 5
TIME_TO_ROLLOUT = 1

class BaseAgent(metaclass=ABCMeta):

    @abstractmethod
    def sample_action(self, state):
        pass


class RandomAgent(BaseAgent):
    def __init__(self, action_space, num_actions=1, seed=None):
        self.action_space = action_space
        self.num_actions = num_actions
        if seed is not None:
            self.action_space.seed(seed)

    def sample_action(self, state=None):
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
        #FIXME: WHAT IS SEED USED FOR?
        if seed is not None:
            self.action_space.seed(seed)

    def sample_action(self, state):
        s = self.action_space.sample()
        action = {}
        selected_actions = self.__GetActionWithMCTS(state)
        #FIXME - what we need the for loop for?
        for sample in selected_actions:
            if isinstance(self.action_space[sample], gym.spaces.Box):
                action[sample] = s[sample][0].item()
            elif isinstance(self.action_space[sample], gym.spaces.Discrete):
                action[sample] = s[sample]
        return action
    
    def __GetActionWithMCTS(self, state):
        #NOTE - flow-on-link___l<i>__t<j> => how many cars in link <i> at time <j>, when 0<=i<=7, 0<=j<=20
        #NOTE - q__l<i>__l<j> =>how many cars are waiting in the queue (t0) to go from link <i> to link <j>, when 0<=i<=7, 0<=j<=7
        #NOTE - Nc___l<i> => how many cars are in link <i> (sum all times), when 0<=i<=7
        #NOTE - virtual-q___l<i> => 
        #NOTE - signal___i0 => the current signl state of i0, when 0<=i<=7
        #NOTE - signal-t___i0 => the time that the intersection i0 has been in the current signal state, when 0<=i<=7
        #NOTE - len() => 
        return random.sample(list(state), self.num_actions)

class MonteCarloTreeSearchNode():
    def __init__(self, state, action_space, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.number_of_visits = 0
        self.score = 0
        self.untried_actions = action_space
        self.action_space = action_space
        return
    
    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.step(action)
        child_node = MonteCarloTreeSearchNode(next_state, self.action_space, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node 
    
    def rollout(self):
        current_rollout_state = self.state
        time_left = TIME_TO_ROLLOUT
        while time_left > 0:
            possible_moves = self.action_space
            
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.step(action)
        return current_rollout_state.score
    
    def rollout_policy(self, possible_moves):
        return random.choice(possible_moves)