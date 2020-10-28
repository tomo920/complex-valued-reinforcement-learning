import numpy as np
import argparse
import os
import datetime
import importlib

from learn import learning
from agent import Agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set parameters.')
    parser.add_argument('env_name', type=str,
                        help='Environment to solve'
                             '[partially observable maze]'
                             '[acrobot]'
                             '[partially observable acrobot]'
                             '[mountain car]'
                             '[partially observable mountain car]',
                        choices=['po_maze', 'acrobot', 'po_acrobot', 'mountain_car', 'po_mountain_car'])
    parser.add_argument('algorithm', type=str,
                        help='Algorithm'
                             '[Q-learning(table)]'
                             '[Qdot-learning(table)]'
                             '[Q-learning(RBF)]'
                             '[Qdot-learning(RBF)]'
                             '[Qdot-learning(NN)]'
                             '[Random]',
                        choices=['q_learning', 'qdot_learning', 'q_learning_rbf', 'qdot_learning_rbf', 'qdot_learning_nn', 'random'])
    parser.add_argument('gamma', type=float, help='Discount factor')
    parser.add_argument('max_steps', type=int, help='Max number of steps in one episode')
    parser.add_argument('--goal_reward', default=100, type=float, help='Positive reward gotten when reaching the goal')
    parser.add_argument('--n_equal_part', default=10, type=int, help='Number of parts of the observation divided when using the table method')
    parser.add_argument('--seed', default=123, type=int, help='Random seed for numpy')
    parser.add_argument('--n_epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--n_episodes', default=1000, type=int, help='Number of episodes in one epoch')
    parser.add_argument('--policy_type', default='boltzmann', type=str,
                        help='Policy used in action selection'
                             '[epsilon_greedy]'
                             '[boltzmann]',
                        choices=['epsilon_greedy', 'boltzmann'])
    parser.add_argument('--boltzmann_t', default=0.5, type=float, help='Boltzmann temperatune in boltzmann policy')
    parser.add_argument('--epsilon_start', default=1.0, type=float, help='Start epsilon of epsilon greedy policy')
    parser.add_argument('--epsilon_end', default=0, type=float, help='End epsilon of epsilon greedy policy')
    parser.add_argument('--epsilon_decay_steps', default=400, type=int,
                        help='Number of steps to decrease epsilon from start to end epsilon greedy policy')
    parser.add_argument('--rotation_angle', default=1, type=float, help='Rotation angle of phase in updating (degree)'
                                                                        '[1]'
                                                                        '[30]'
                                                                        '[180]')
    parser.add_argument('--trace_num', default=6, type=int, help='Trace number of eligibility trace')
    parser.add_argument('--alpha', default=0.001, type=float, help='Learning rate for table method')
    parser.add_argument('--lr_h', default=0.0001, type=float, help='Learning rate of hidden layer of network')
    parser.add_argument('--lr_o', default=0.001, type=float, help='Learning rate of output layer of network')
    parser.add_argument('--action_dim', default=1, type=int, help='Action dimension of environment')
    parser.add_argument('--hidden_size', default=30, type=int, help='Number of neurons in hidden layer of network')
    parser.add_argument('--is_t_change', action='store_true')
    parser.add_argument('--is_all_positive_action', action='store_true')
    parser.add_argument('--is_legal_action', action='store_true')
    parser.add_argument('--is_continuous', action='store_true')
    parser.add_argument('--is_special_quantization', action='store_true')

    config = parser.parse_args()

    v_config = vars(config)
    config_text = ''
    for k in v_config:
        config_text += '{0}: {1}\n'.format(k, v_config[k])
    print(config_text)

    np.random.seed(config.seed)

    save_path = os.path.abspath("./result/{}_result".format(config.env_name))
    save_dir = os.path.join(save_path, "{0}_{1}".format(config.algorithm, datetime.datetime.now().isoformat()))
    os.makedirs(save_dir)
    with open('{}/config.txt'.format(save_dir), mode='w') as f:
        f.write(config_text)

    # make Environment
    env_module = importlib.import_module('envs.{}'.format(config.env_name))
    if config.env_name.split('_')[0] == 'po':
        env = env_module.PoEnv(config)
    else:
        env = env_module.Env(config)

    # make Agent
    agent = Agent(config, env)

    for i in range(config.n_epochs):
        agent.reset_q_func()
        learning(config,
                 env,
                 agent,
                 i,
                 save_dir)
