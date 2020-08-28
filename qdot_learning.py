import numpy as np
import math
import cmath

task = 'po_acrobot' #'po_maze' or 'po_acrobot'

if task == 'po_maze':
    from pomaze_env import Env, legal_states, max_step, legal_action, legal_action_list, observation_size, action_size
    epsilon_start = 1.0
    epsilon_end = 0
    epsilon_decay_steps = 50000
    episode_num = 500000
    save_period = 10
    update_num = 10
    beta = cmath.rect(1, math.pi/6.0)
    alpha = 0.25
    gamma = 0.9
    Ne = 1
    policy_type = 'boltzmann'
    T = 20.0
elif task == 'po_acrobot':
    from po_acrobot import Env, max_step, legal_action, observation_size, action_size
    epsilon_start = 1.0
    epsilon_end = 0
    epsilon_decay_steps = 50000
    episode_num = 500000
    save_period = 10
    update_num = 10
    beta = cmath.rect(1, math.pi/180.0)
    alpha = 0.15
    gamma = 0.9
    Ne = 6
    # Ne = 1
    policy_type = 'boltzmann'
    T = 10.0

result = []

class QdotLearning(): #o
    """
    Q learning class.
    Q value is the complex function of observation and action.
    """

    def __init__(self):
        self.env = Env()
        # initialize Q table
        self.Q = {}
        for observation in range(1, observation_size+1):
            if legal_action:
                self.Q[observation] = {}
                for a in legal_action_list[observation]:
                    self.Q[observation][a] = complex(0, 0)
            else:
                self.Q[observation] = [complex(0, 0) for _ in range(action_size)]

    def get_policy(self, observation, epsilon):
        if policy_type == 'epsilon greedy':
            '''
            epsilon greedy policy
            '''
            if legal_action:
                a_size = len(legal_action_list[observation])
            else:
                a_size = action_size
            pi = np.ones(a_size, dtype='float32') * epsilon / a_size
            max_action = self.get_max_action(observation)
            pi[max_action] += 1.0-epsilon
            return pi
        elif policy_type == 'boltzmann':
            '''
            boltzmann policy
            '''
            if legal_action:
                e_Q = np.exp([(Q * self.I.conjugate()).real / T for Q in self.Q[observation].values()])
            else:
                e_Q = np.exp([(Q * self.I.conjugate()).real / T for Q in self.Q[observation]])
            return e_Q / np.sum(e_Q)

    def get_max_action(self, observation):
        if legal_action:
            return np.argmax([(Q * self.I.conjugate()).real for Q in self.Q[observation].values()])
        else:
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
            if legal_action:
                a_index = np.argmax([abs(Q) for Q in self.Q[observation].values()])
                a = legal_action_list[observation][a_index]
            else:
                a = np.argmax([abs(Q) for Q in self.Q[observation]])
            self.I = self.Q[observation][a]
            # initialize history
            self.o_history = []
            self.a_history = []
            epsilon = epsilons[min(i, epsilon_decay_steps-1)]
            while True:
                pi = self.get_policy(observation, epsilon)
                if legal_action:
                    action = np.random.choice(legal_action_list[observation], p = pi)
                else:
                    action = np.random.choice(action_size, p = pi)
                next_observation, reward, done = self.env.step(action)
                if self.env.steps > max_step:
                    result.append(self.env.steps)
                    break
                # update history
                self.o_history.append(observation)
                self.a_history.append(action)
                # update internal reference valuue
                self.update_I(observation, action)
                # update Q table
                self.update_q(next_observation, reward, done)
                # update internal reference value
                self.update_I(observation, action)
                if done:
                    result.append(self.env.steps)
                    break
                else:
                    observation = next_observation

    def update_q(self, next_observation, reward, done):
        if legal_action:
            a_index = self.get_max_action(next_observation)
            a = legal_action_list[next_observation][a_index]
        else:
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
