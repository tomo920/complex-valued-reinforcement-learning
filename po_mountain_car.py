import gym
import numpy as np
from numpy import sin, cos, pi
import math

is_continuous = False
if is_continuous:
    observation_size = 1
else:
    observation_size = 1700

legal_action = False
action_size = 3
action_list = [-1.0, 0, 1.0]

max_step = 10000

max_position = 0.5
min_position = -1.2
max_velocity = 0.07
min_velocity = -0.07

o_n = np.linspace(min_position, max_position, observation_size+1)

class Env():
    def __init__(self):
        self.reset()

    def discret(self, position):
        for n in range(1, observation_size+1):
            if position <= o_n[n]:
                return n

    def get_observation(self):
        if is_continuous:
            return self.position
        else:
            return self.discret(self.position)

    def reset(self):
        self.position = -0.5
        self.velocity = 0
        self.steps = 0
        return self.get_observation()

    def step(self, action):
        self.steps += 1
        self.velocity += 0.001 * action_list[action] - 0.0025 * math.cos(self.position*3.0)
        self.velocity = np.clip(self.velocity, min_velocity, max_velocity)
        self.position += self.velocity
        self.position = np.clip(self.position, min_position, max_position)
        if self.position >= 0.5:
            reward = 100.0
            done = True
        else:
            reward = 0.0
            done = False
        return self.get_observation(), reward, done
