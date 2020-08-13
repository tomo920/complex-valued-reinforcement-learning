import os
from contextlib import redirect_stderr
from copy import deepcopy

import gym
import numpy as np

'''
・・・・・・・
・・14151617・
・1112・13・・
・７・８９10・
・４５・６・・
・・９２11４・
・・・・・・・
'''
legal_states = [np.array([2.0, 1.0]), np.array([3.0, 1.0]), np.array([4.0, 1.0]), np.array([5.0, 1.0]),
                np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([4.0, 2.0]),
                np.array([1.0, 3.0]), np.array([3.0, 3.0]), np.array([4.0, 3.0]), np.array([5.0, 3.0]),
                np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([4.0, 4.0]),
                np.array([2.0, 5.0]), np.array([3.0, 5.0]), np.array([4.0, 5.0]), np.array([5.0, 5.0])]
illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]), np.array([6.0, 0.0]),
                  np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([6.0, 1.0]),
                  np.array([0.0, 2.0]), np.array([3.0, 2.0]), np.array([5.0, 2.0]), np.array([6.0, 2.0]),
                  np.array([0.0, 3.0]), np.array([2.0, 3.0]), np.array([6.0, 3.0]),
                  np.array([0.0, 4.0]), np.array([3.0, 4.0]), np.array([5.0, 4.0]), np.array([6.0, 4.0]),
                  np.array([0.0, 5.0]), np.array([1.0, 5.0]), np.array([6.0, 5.0]),
                  np.array([0.0, 6.0]), np.array([1.0, 6.0]), np.array([2.0, 6.0]), np.array([3.0, 6.0]), np.array([4.0, 6.0]), np.array([5.0, 6.0]), np.array([6.0, 6.0])]
legal_states = [state.tostring() for state in legal_states]
illegal_states = [state.tostring() for state in illegal_states]

observation_list = {}
observation_list[legal_states[0]] = 9
observation_list[legal_states[1]] = 2
observation_list[legal_states[2]] = 11
observation_list[legal_states[3]] = 4
observation_list[legal_states[4]] = 9
observation_list[legal_states[5]] = 10
observation_list[legal_states[6]] = 6
observation_list[legal_states[7]] = 6
observation_list[legal_states[8]] = 7
observation_list[legal_states[9]] = 8
observation_list[legal_states[10]] = 4
observation_list[legal_states[11]] = 1
observation_list[legal_states[12]] = 5
observation_list[legal_states[13]] = 6
observation_list[legal_states[14]] = 1
observation_list[legal_states[15]] = 2
observation_list[legal_states[16]] = 3
observation_list[legal_states[17]] = 4

start_state = np.array([1.0, 3.0])
goal_state = np.array([5.0, 3.0])


'''
action 0 -> right
action 1 -> left
action 2 -> up
action 3 -> down
'''
action_list = [np.array([1.0, 0.0]),
               np.array([-1.0, 0.0]),
               np.array([0.0, 1.0]),
               np.array([0.0, -1.0])]

max_step = 5000

class Env():
    def __init__(self):
        self.reset()

    def get_observation(self):
        state = self.state.tostring()
        return observation_list[state]

    def check_legal(self):
        state = self.state.tostring()
        if state in legal_states:
            return True
        else:
            return False

    def reset(self):
        self.state = start_state
        self.steps = 0
        return self.get_observation()

    def step(self, action):
        self.steps += 1
        c_state = self.state
        self.state = self.state+action_list[action]
        if not self.check_legal():
            self.state = c_state
        if self.state[0] == goal_state[0] and self.state[1] == goal_state[1]:
            reward = 100.0/(self.steps-7.0)
            done = True
        elif self.steps == max_step:
            reward = 0.0
            done = True
        else:
            reward = 0.0
            done = False
        return self.get_observation(), reward, done
