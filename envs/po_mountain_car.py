import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from mountain_car import Env

class PoEnv(Env):
    def __init__(self, config):
        super().__init__(config)
        if config.is_continuous:
            self.observation_size = 1
        else:
            self.observation_size = config.n_equal_part

    def get_observation(self):
        if self.config.is_continuous:
            return np.array([self.position])
        else:
            o1 = self.discret(self.position, self.o1_list)
            return o1
