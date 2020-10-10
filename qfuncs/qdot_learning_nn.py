import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))

from qdot_base import QdotBase
from complexnet import CNnet

class Qfunc(QdotBase):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.action_size = env.action_size
        self.action_list = env.action_list
        # neural network configulation
        hidden_config = {
            'input_size': env.observation_size + 1,
            'output_size': config.hidden_size,
            'activation': 'tanh',
            'lr': config.lr_h
        }
        output_config = {
            'input_size': config.hidden_size,
            'output_size': 1,
            'activation': 'linear',
            'lr': config.lr_o
        }
        layer_config = [hidden_config, output_config]
        # initialize Q network
        self.Q_network = CNnet(layer_config)

    def get_q_o_a(self, observation, action):
        action = np.array([self.action_list[action]])
        input = np.concatenate([observation, action])
        return self.Q_network.outputs(input.astype(np.complex128))

    def get_q_o(self, observation):
        return np.concatenate([self.get_q_o_a(observation, action) for action in range(self.action_size)])

    def update_q(self, observation, action, q_target):
        action = np.array([self.action_list[action]])
        input = np.concatenate([observation, action])
        self.Q_network.train(input.astype(np.complex128), q_target, 'square_error')
