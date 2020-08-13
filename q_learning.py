import numpy as np

from pomaze_env import Env, legal_states, max_step

action_size = 4
epsilon_start = 1.0
epsilon_end = 0
epsilon_decay_steps = 50000
episode_num = 500000
save_period = 10
update_num = 10
alpha = 0.1
gamma = 0.9

result = []

class QLearning(): #o
    """
    Q learning class.
    Q value is the function of observation and action
    """

    def __init__(self):
        self.env = Env()
        # initialize Q table
        self.Q = {}
        for observation in range(1, 12):
            self.Q[observation] = np.zeros(action_size)

    def get_policy(self, observation, epsilon):
        '''
        epsilon greedy policy
        '''
        pi = np.ones(action_size, dtype='float32') * epsilon / action_size
        q = self.Q[observation]
        max_action = np.argmax(q)
        pi[max_action] += 1.0-epsilon
        return pi

    def learn(self):
        epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # start learning
        for i in range(episode_num):
            if i % save_period == 0:
                np.save('result', result)
            # execute episode
            observation = self.env.reset()
            epsilon = epsilons[min(i, epsilon_decay_steps-1)]
            while True:
                pi = self.get_policy(observation, epsilon)
                action = np.random.choice(action_size, p = pi)
                next_observation, reward, done = self.env.step(action)
                if self.env.steps == max_step:
                    break
                # update Q table
                self.update_q(observation, action, next_observation, reward, done)
                if done:
                    result.append(reward)
                    break
                else:
                    observation = next_observation

    def update_q(self, observation, action, next_observation, reward, done):
        self.Q[observation][action] = self.Q[observation][action] + alpha*(reward+gamma*np.max(self.Q[next_observation])*(1.0-done) - self.Q[observation][action])
