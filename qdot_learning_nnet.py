import numpy as np
import math
import cmath
from complexnet import CNnet

task = 'po_acrobot' #'po_maze' or 'po_acrobot' or 'po_mountain_car' or 'mountain_car' or 'acrobot'

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
    episode_num = 50000
    save_period = 10
    update_num = 10
    gamma = 0.7
    policy_type = 'boltzmann'
    T = 0.5
    beta = cmath.rect(1, math.pi/180.0)
    Ne = 1
    lr_h = 0.0001
    lr_o = 0.001
    action_dim = 1
    hidden_size = 30
    all_positive_action = False
elif task == 'po_mountain_car':
    from po_mountain_car import Env, max_step, legal_action, observation_size, action_size
    epsilon_start = 1.0
    epsilon_end = 0
    epsilon_decay_steps = 50000
    episode_num = 50000
    save_period = 10
    update_num = 10
    gamma = 0.7
    policy_type = 'boltzmann'
    T = 0.5
    beta = cmath.rect(1, math.pi/180.0)
    Ne = 1
    lr_h = 0.0001
    lr_o = 0.001
    action_dim = 1
    hidden_size = 30
    all_positive_action = False

if all_positive_action:
    action_list = [action * 5.0 + 5.0 for action in action_list]

hidden_config = {
    'input_size': observation_size + action_dim,
    'output_size': hidden_size,
    'activation': 'tanh',
    'lr': lr_h
}

output_config = {
    'input_size': hidden_size,
    'output_size': 1,
    'activation': 'linear',
    'lr': lr_o
}

layer_config = [hidden_config, output_config]

result = []

class QdotLearning_Nnet(): #o
    """
    Q learning class.
    Q value is expressed by complex valued neural network whose input is observation and action.
    """

    def __init__(self):
        self.env = Env()
        # initialize Q network
        self.Q_network = CNnet(layer_config)

    def get_q_o_a(self, observation, action):
        action = np.array([self.env.action_list[action]])
        input = np.concatenate([observation, action])
        return self.Q_network.outputs(input.astype(np.complex128))

    def get_q(self, observation):
        return np.concatenate([self.get_q_o_a(observation, action) for action in range(action_size)])

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
                Q = [(Q * self.I.conjugate()).real / T for Q in q]
                Q_max = np.max(Q)
                e_Q_imp = np.exp(Q-Q_max)
            return e_Q_imp / np.sum(e_Q_imp)

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
            action = np.array([self.env.action_list[action]])
            input = np.concatenate([observation, action])
            target = next_q * beta**(k+1)
            self.Q_network.train(input.astype(np.complex128), target, 'square_error')

    def update_I(self, observation, action):
        q = self.get_q(observation)
        self.I = q[action] / beta
