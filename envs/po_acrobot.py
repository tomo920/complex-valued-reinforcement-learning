import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from acrobot import Env

class PoEnv(Env):
    def __init__(self, config):
        super().__init__(config)
        if config.is_continuous:
            self.observation_size = 2
        else:
            self.observation_size = config.n_equal_part**2

    def get_observation(self):
        theta1 = self.env.env.state[0]
        theta2 = self.env.env.state[1]
        if self.config.is_continuous:
            return np.array([theta1, theta2])
        else:
            o1 = self.discret(theta1, self.o1_list)
            o2 = self.discret(theta2, self.o2_list)
            return o1 + 10 * (o2 - 1)
