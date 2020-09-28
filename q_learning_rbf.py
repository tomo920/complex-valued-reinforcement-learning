import numpy as np
from rbfnet import RBFNet

task = 'mountain_car' #'po_maze' or 'po_acrobot' or 'po_mountain_car' or 'mountain_car' or 'acrobot'

if task == 'po_maze':
    from pomaze_env import Env, legal_states, max_step, legal_action, legal_action_list, observation_size, action_size
    epsilon_start = 1.0
    epsilon_end = 0
    epsilon_decay_steps = 50000
    episode_num = 500
    save_period = 10
    update_num = 10
    gamma = 0.9
    policy_type = 'boltzmann'
    T_init = 100.0
elif task == 'po_acrobot':
    from po_acrobot import Env, max_step, legal_action, observation_size, action_size
    epsilon_start = 1.0
    epsilon_end = 0
    epsilon_decay_steps = 50000
    episode_num = 500000
    save_period = 10
    update_num = 10
    gamma = 0.9
    policy_type = 'boltzmann'
    T = 0.1
    # T = 10.0
    weight_lr = 0.001
    mu_lr = 0.001
    sigma_lr = 0.001
    hidden_size = 3
elif task == 'po_mountain_car':
    from po_mountain_car import Env, max_step, legal_action, observation_size, action_size
    epsilon_start = 1.0
    epsilon_end = 0
    epsilon_decay_steps = 50000
    episode_num = 300
    save_period = 10
    update_num = 10
    gamma = 0.7
    policy_type = 'boltzmann'
    T_init = 150.0
    weight_lr = 0.001
    mu_lr = 0.001
    sigma_lr = 0.001
    hidden_size = 3
elif task == 'mountain_car':
    from mountain_car import Env, max_step, legal_action, observation_size, action_size
    epsilon_start = 1.0
    epsilon_end = 0
    epsilon_decay_steps = 50000
    episode_num = 5000
    save_period = 10
    update_num = 10
    gamma = 0.7
    policy_type = 'boltzmann'
    T_init = 150.0
    T = 1.0
    weight_lr = 0.001
    mu_lr = 0.001
    sigma_lr = 0.001
    hidden_size = 10
elif task == 'acrobot':
    from acrobot import Env, max_step, legal_action, observation_size, action_size
    epsilon_start = 1.0
    epsilon_end = 0
    epsilon_decay_steps = 50000
    episode_num = 1000
    save_period = 10
    update_num = 10
    gamma = 0.7
    policy_type = 'boltzmann'
    T_init = 150.0
    weight_lr = 0.01
    mu_lr = 0.01
    sigma_lr = 0.01
    hidden_size = 20

result = []

class QLearning_RBF(): #o
    """
    Q learning with RBF Network class.
    Q value is represented by RBF Networks which take observatin as input and outputs Q(o, a).
    """

    def __init__(self):
        self.env = Env()
        # initialize Q network
        self.Q_network = []
        for _ in range(action_size):
            self.Q_network.append(RBFNet(observation_size, hidden_size, weight_lr, mu_lr, sigma_lr))

    def get_q(self, observation):
        return np.array([q_net.outputs(observation) for q_net in self.Q_network])

    def get_policy(self, observation, epsilon):
        if policy_type == 'epsilon greedy':
            '''
            epsilon greedy policy
            '''
            q = self.get_q(observation)
            if legal_action:
                a_size = len(legal_action_list[observation])
                # masking
            else:
                a_size = action_size
            pi = np.ones(a_size, dtype='float32') * epsilon / a_size
            max_action = np.argmax(q)
            pi[max_action] += 1.0-epsilon
            return pi
        elif policy_type == 'boltzmann':
            '''
            boltzmann policy
            '''
            q = self.get_q(observation)
            if legal_action:
                # masking
                q = self.get_q(observation)
            e_Q = np.exp(q / self.T)
            return e_Q / np.sum(e_Q)

    def learn(self):
        epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # start learning
        for i in range(episode_num):
            if i % save_period == 0:
                np.save('result', result)
            # execute episode
            observation = self.env.reset()
            if task == 'po_maze':
                # update T
                self.T = T_init / (1 + i)
            elif task == 'po_acrobot':
                self.T = T
            elif task == 'po_mountain_car':
                self.T = T_init / (1 + i)
            elif task == 'mountain_car':
                # self.T = T_init / (1 + i)
                self.T = T
            elif task == 'acrobot':
                self.T = T_init / (1 + i)
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
                # update Q network
                self.update_q(next_observation, reward, done)
                if done:
                    result.append(self.env.steps)
                    break
                else:
                    observation = next_observation

    def update_q(self, next_observation, reward, done):
        q = self.get_q(next_observation)
        if legal_action:
            # masking
            q = self.get_q(next_observation)
        next_q = reward + gamma * np.max(q) * (1.0-done)
        observation = self.o_history[-1]
        action = self.a_history[-1]
        td_error = reward + gamma * np.max(q) * (1.0-done) - self.get_q(observation)[action]
        self.Q_network[action].update(observation, td_error)
