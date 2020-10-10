import numpy as np
from numpy import pi
import cmath

import importlib

class Learner():
    '''
    Learner class.
    Algorithm is Q-learning(table) or Qdot-learning(table) or
                 Q-learning(RBF) or Qdot-learning(RBF) or
                 Qdot-learning(NN) or Random
    '''

    def __init__(self, config, i_epoch, save_dir):
        self.config = config
        self.i_epoch = i_epoch
        self.save_dir = save_dir
        # initialize environment
        env_module = importlib.import_module('envs.{}'.format(config.env_name))
        if config.env_name.split('_')[0] == 'po':
            self.env = env_module.PoEnv(config)
        else:
            self.env = env_module.Env(config)
        # initialize Q function
        alg_module = importlib.import_module('qfuncs.{}'.format(config.algorithm))
        self.q_func = alg_module.Qfunc(config, self.env)
        self.alg_type = config.algorithm.split('_')[0]
        if self.alg_type == 'qdot':
            self.beta = cmath.rect(1, pi/config.rotation_angle)

    def get_policy(self, observation, i_episode):
        if self.alg_type == 'qdot':
            q = self.q_func.get_effective_q(observation, self.ir_value)
        else:
            q = self.q_func.get_effective_q(observation)
        if self.config.policy_type == 'epsilon_greedy':
            '''
            epsilon greedy policy
            '''
            epsilon = self.epsilons[min(i_episode, self.config.epsilon_decay_steps-1)]
            if self.config.is_legal_action:
                a_size = len(self.env.legal_action_list[observation])
            else:
                a_size = self.env.action_size
            pi = np.ones(a_size, dtype='float32') * epsilon / a_size
            max_action = np.argmax(q)
            pi[max_action] += 1.0-epsilon
            return pi
        elif self.config.policy_type == 'boltzmann':
            '''
            boltzmann policy
            '''
            if self.config.is_t_change:
                T = self.config.boltzmann_t / (1 + i_episode)
            else:
                T = self.config.boltzmann_t
            Q = [Q / T for Q in q]
            Q_max = np.max(Q)
            e_Q_imp = np.exp(Q-Q_max)
            return e_Q_imp / np.sum(e_Q_imp)

    def update_ir_value(self, observation, action):
        self.ir_value = self.q_func.get_q_o(observation)[action] / self.beta

    def learn(self):
        result = []

        if self.config.policy_type == 'epsilon_greedy':
            self.epsilons = np.linspace(self.config.epsilon_start, self.config.epsilon_end, self.config.epsilon_decay_steps)

        # start learning
        for i in range(self.config.n_episodes):
            # execute episode
            observation = self.env.reset()
            if self.alg_type == 'qdot':
                # initialize internal reference value
                q = self.q_func.get_q(observation)
                a = np.argmax([abs(Q) for Q in q])
                if self.config.is_legal_action:
                    a = self.env.legal_action_list[observation][a]
                self.ir_value = q[a]
            # initialize history
            self.o_history = []
            self.a_history = []
            while True:
                pi = self.get_policy(observation, i)
                if self.config.is_legal_action:
                    action = np.random.choice(self.env.legal_action_list[observation], p = pi)
                else:
                    action = np.random.choice(self.env.action_size, p = pi)
                next_observation, reward, done = self.env.step(action)
                # update history
                self.o_history.append(observation)
                self.a_history.append(action)
                if self.alg_type == 'qdot':
                    # update internal reference valuue
                    self.update_ir_value(observation, action)
                # update Q network
                self.update(next_observation, reward, done)
                if self.alg_type == 'qdot':
                    # update internal reference valuue
                    self.update_ir_value(observation, action)
                if done:
                    result.append(self.env.steps)
                    np.save('{0}/result_{1}.npy'.format(self.save_dir, self.i_epoch), result)
                    break
                else:
                    observation = next_observation

    def update(self, next_observation, reward, done):
        if self.alg_type == 'qdot':
            effective_q_ = self.q_func.get_effective_q(next_observation, self.ir_value)
            max_index = np.argmax(effective_q_)
            max_q_ = self.q_func.get_q(next_observation)[max_index]
        else:
            q_ = self.q_func.get_effective_q(next_observation)
            max_q_ = np.max(q_)
        q_target = reward + self.config.gamma * max_q_ * (1.0-done)
        for k in range(self.config.trace_num):
            if len(self.o_history) < k+1:
                break
            observation = self.o_history[-(k+1)]
            action = self.a_history[-(k+1)]
            if self.alg_type == 'qdot':
                q_target *= self.beta
            self.q_func.update_q(observation, action, q_target)
