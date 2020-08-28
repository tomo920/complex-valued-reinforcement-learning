import numpy as np

task = 'po_acrobot' #'po_maze' or 'po_acrobot'

if task == 'po_maze':
    from pomaze_env import Env, legal_states, max_step, legal_action, legal_action_list, observation_size, action_size
    epsilon_start = 1.0
    epsilon_end = 0
    epsilon_decay_steps = 50000
    episode_num = 500
    save_period = 10
    update_num = 10
    alpha_init = 0.001
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
    alpha = 0.25
    gamma = 0.9
    policy_type = 'boltzmann'
    T = 0.1
    # T = 10.0

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
        for observation in range(1, observation_size+1):
            if legal_action:
                self.Q[observation] = {}
                for a in legal_action_list[observation]:
                    self.Q[observation][a] = 0.0
            else:
                self.Q[observation] = np.zeros(action_size)

    def get_policy(self, observation, epsilon):
        if policy_type == 'epsilon greedy':
            '''
            epsilon greedy policy
            '''
            if legal_action:
                a_size = len(legal_action_list[observation])
                q = self.Q[observation].values()
            else:
                a_size = action_size
                q = self.Q[observation]
            pi = np.ones(a_size, dtype='float32') * epsilon / a_size
            max_action = np.argmax(q)
            pi[max_action] += 1.0-epsilon
            return pi
        elif policy_type == 'boltzmann':
            '''
            boltzmann policy
            '''
            if legal_action:
                e_Q = np.exp([Q / self.T for Q in self.Q[observation].values()])
            else:
                e_Q = np.exp([Q / self.T for Q in self.Q[observation]])
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
                # update alpha
                self.alpha = alpha_init * (episode_num - i)
                # update T
                self.T = T_init / (1 + i)
            elif task == 'po_acrobot':
                self.alpha = alpha
                self.T = T
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
                # update Q table
                self.update_q(next_observation, reward, done)
                if done:
                    result.append(self.env.steps)
                    break
                else:
                    observation = next_observation

    def update_q(self, next_observation, reward, done):
        if legal_action:
            q = [Q for Q in self.Q[next_observation].values()]
        else:
            q = self.Q[next_observation]
        next_q = reward + gamma * np.max(q) * (1.0-done)
        observation = self.o_history[-1]
        action = self.a_history[-1]
        self.Q[observation][action] = self.Q[observation][action] \
            + self.alpha * (next_q - self.Q[observation][action])
