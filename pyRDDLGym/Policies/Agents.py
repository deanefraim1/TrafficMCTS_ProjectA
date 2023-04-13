from abc import ABCMeta, abstractmethod
import random

import gym


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
        #NOTE - signal___i0 => 
        #NOTE - signal-t___i0 =>
        #NOTE - len() => 
        return random.sample(list(state), self.num_actions)

