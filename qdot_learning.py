import numpy as np
import math
import cmath

from pomaze_env import Env, legal_states, max_step

action_size = 4
epsilon_start = 1.0
epsilon_end = 0
epsilon_decay_steps = 50000
episode_num = 500000
save_period = 10
update_num = 10
beta = cmath.rect(1, math.pi/6.0)
alpha = 0.1
gamma = 0.9
Ne = 1

result = []

class QLearning(): #o
    """
    Q learning class.
    Q value is the complex function of observation and action.
    """

    def __init__(self):
        self.env = Env()
        # initialize Q table
        self.Q = {}
        for observation in range(1, 12):
            self.Q[observation] = [complex(0, 0) for _ in range(action_size)]

    def get_policy(self, observation, epsilon):
        '''
        epsilon greedy policy
        '''
        pi = np.ones(action_size, dtype='float32') * epsilon / action_size
        max_action = self.get_max_action(observation)
        pi[max_action] += 1.0-epsilon
        return pi

    def get_max_action(self, observation):
        return np.argmax([(Q * self.I.conjugate()).real for Q in self.Q[observation]])

    def learn(self):
        epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # start learning
        for i in range(episode_num):
            if i % save_period == 0:
                np.save('result', result)
            # execute episode
            observation = self.env.reset()
            # initialize internal reference value
            a = np.argmax([abs(Q) for Q in self.Q[observation]])
            self.I = self.Q[observation][a]
            # initialize history
            self.o_history = []
            self.a_history = []
            epsilon = epsilons[min(i, epsilon_decay_steps-1)]
            while True:
                pi = self.get_policy(observation, epsilon)
                action = np.random.choice(action_size, p = pi)
                next_observation, reward, done = self.env.step(action)
                if self.env.steps == max_step:
                    break
                # update history
                self.o_history.append(observation)
                self.a_history.append(action)
                # update Q table
                self.update_q(next_observation, reward, done)
                # update internal reference value
                self.update_I(observation, action)
                if done:
                    result.append(reward)
                    break
                else:
                    observation = next_observation

    def update_q(self, next_observation, reward, done):
        a = self.get_max_action(next_observation)
        next_q = reward + gamma * self.Q[next_observation][a] * (1.0-done)
        for k in range(Ne):
            if len(self.o_history) < k+1:
                break
            observation = self.o_history[-(k+1)]
            action = self.a_history[-(k+1)]
            self.Q[observation][action] = self.Q[observation][action] \
                + alpha * (next_q * beta**(k+1) - self.Q[observation][action])

    def update_I(self, observation, action):
        self.I = self.Q[observation][action] / beta
