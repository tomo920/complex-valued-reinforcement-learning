import os
import sys
sys.path.append(os.path.dirname(__file__))

from q_base import QBase

class QdotBase(QBase):
    def __init__(self, config, env):
        super().__init__(config, env)

    def get_effective_q(self, observation, ir_value):
        q = self.get_q(observation)
        return [(Q * ir_value.conjugate()).real for Q in q]
