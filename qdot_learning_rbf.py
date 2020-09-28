import numpy as np
import math
import cmath
from rbfnet import RBFNet

task = 'po_mountain_car' #'po_maze' or 'po_acrobot' or 'po_mountain_car' or 'mountain_car' or 'acrobot'

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
elif task == 'po_mountain_car':
    from po_mountain_car import Env, max_step, legal_action, observation_size, action_size
    epsilon_start = 1.0
    epsilon_end = 0
    epsilon_decay_steps = 50000
    episode_num = 5000
    save_period = 10
    update_num = 10
    gamma = 0.7
    policy_type = 'boltzmann'
    T = 0.5
    beta = cmath.rect(1, math.pi/180.0)
    Ne = 1
    weight_lr = 0.01
    mu_lr = 0.001
    sigma_lr = 0.001
    hidden_size = 5

result = []

class ComplexRBFNet():
    """
    RBF network class for representing complex value
    """

    def __init__(self, input_size, hidden_size, weight_lr, mu_lr, sigma_lr):
        self.real_part = RBFNet(input_size, hidden_size, weight_lr, mu_lr, sigma_lr)
        self.imaginary_part = RBFNet(input_size, hidden_size, weight_lr, mu_lr, sigma_lr)

    def outputs(self, input):
        real = self.real_part.outputs(input)
        imaginary = self.imaginary_part.outputs(input)
        return complex(real, imaginary)

    def update(self, input, td_error):
        error_real = td_error.real
        error_imag = td_error.imag
        self.real_part.update(input, error_real)
        self.imaginary_part.update(input, error_imag)

class QdotLearning_RBF(): #o
    """
    Q learning class.
    Q value is the complex function of observation and action.
    """

    def __init__(self):
        self.env = Env()
        # initialize Q network
        self.Q_network = []
        for _ in range(action_size):
            self.Q_network.append(ComplexRBFNet(observation_size, hidden_size, weight_lr, mu_lr, sigma_lr))

    def get_q(self, observation):
        return np.array([q_net.outputs(observation) for q_net in self.Q_network])

    def get_policy(self, observation, epsilon):
        if policy_type == 'epsilon greedy':
            '''
            epsilon greedy policy
            '''
            if legal_action:
                a_size = len(legal_action_list[observation])
                # masking
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
            q = self.get_q(observation)
            if legal_action:
                e_Q = np.exp([(Q * self.I.conjugate()).real / T for Q in q])
                # masking
            else:
                e_Q = np.exp([(Q * self.I.conjugate()).real / T for Q in q])
            return e_Q / np.sum(e_Q)

    def get_max_action(self, observation):
        q = self.get_q(observation)
        if legal_action:
            return np.argmax([(Q * self.I.conjugate()).real for Q in q])
            # masking
        else:
            return np.argmax([(Q * self.I.conjugate()).real for Q in q])

    def learn(self):
        epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # start learning
        for i in range(episode_num):
            if i % save_period == 0:
                np.save('result', result)
            # execute episode
            observation = self.env.reset()
            # initialize internal reference value
            q = self.get_q(observation)
            if legal_action:
                a_index = np.argmax([abs(Q) for Q in q])
                a = legal_action_list[observation][a_index]
                # masking
            else:
                a = np.argmax([abs(Q) for Q in q])
            self.I = q[a]
            # initialize history
            self.o_history = []
            self.a_history = []
            epsilon = epsilons[min(i, epsilon_decay_steps-1)]
            while True:
                pi = self.get_policy(observation, epsilon)
                if legal_action:
                    action = np.random.choice(legal_action_list[observation], p = pi)
                    # masking
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
                # update Q network
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
            # masking
        else:
            a = self.get_max_action(next_observation)
        next_q = reward + gamma * self.get_q(next_observation)[a] * (1.0-done)
        for k in range(Ne):
            if len(self.o_history) < k+1:
                break
            observation = self.o_history[-(k+1)]
            action = self.a_history[-(k+1)]
            td_error = next_q * beta**(k+1) - self.get_q(observation)[action]
            self.Q_network[action].update(observation, td_error)

    def update_I(self, observation, action):
        q = self.get_q(observation)
        self.I = q[action] / beta
