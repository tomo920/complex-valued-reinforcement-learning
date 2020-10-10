import numpy as np
from numpy import cos
import sys
import os
sys.path.append(os.path.dirname(__file__))

from env_base import EnvBase

action_list = [-1.0, 0, 1.0]

max_position = 0.5
min_position = -1.2
max_velocity = 0.07
min_velocity = -0.07

class Env(EnvBase):
    def __init__(self, config):
        if config.is_continuous:
            observation_size = 2
        else:
            observation_size = config.n_equal_part**2
            self.o1_list = np.linspace(min_position, max_position, config.n_equal_part+1)
            self.o2_list = np.linspace(min_velocity, max_velocity, config.n_equal_part+1)
        action_size = 3
        max_step = config.max_steps
        super().__init__(config, observation_size, action_size, max_step, action_list)

    def get_observation(self):
        if self.config.is_continuous:
            return np.array([self.position, self.velocity])
        else:
            o1 = self.discret(self.position, self.o1_list)
            o2 = self.discret(self.velocity, self.o2_list)
            return o1 + 10 * (o2 - 1)

    def reset_state(self):
        self.position = -0.5
        self.velocity = 0

    def change_state(self, action):
        self.velocity += 0.001 * action_list[action] - 0.0025 * cos(self.position*3.0)
        self.velocity = np.clip(self.velocity, min_velocity, max_velocity)
        self.position += self.velocity
        self.position = np.clip(self.position, min_position, max_position)

    def check_goal(self):
        return self.position >= 0.5
