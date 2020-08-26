import numpy as np
import math
import cmath

from pomaze_env import Env, legal_states, max_step, legal_action, legal_action_list

action_size = 4
episode_num = 500
save_period = 10

result = []

class RandomWalk(): #o
    """
    Q learning class.
    Q value is the complex function of observation and action.
    """

    def __init__(self):
        self.env = Env()

    def get_policy(self, observation):
        '''
        random walk
        '''
        if legal_action:
            a_size = len(legal_action_list[observation])
        else:
            a_size = action_size
        pi = np.ones(a_size, dtype='float32') / a_size
        return pi

    def learn(self):
        # start learning
        for i in range(episode_num):
            if i % save_period == 0:
                np.save('result', result)
            # execute episode
            observation = self.env.reset()
            while True:
                pi = self.get_policy(observation)
                if legal_action:
                    action = np.random.choice(legal_action_list[observation], p = pi)
                else:
                    action = np.random.choice(action_size, p = pi)
                next_observation, reward, done = self.env.step(action)
                if self.env.steps > max_step:
                    result.append(self.env.steps)
                    break
                if done:
                    result.append(self.env.steps)
                    break
                else:
                    observation = next_observation
