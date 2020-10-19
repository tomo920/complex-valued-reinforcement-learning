import numpy as np

def learning(config,
             env,
             agent,
             i_epoch,
             save_dir):
    '''
    Execute one epoch of learning.
    '''
    result = []
    # start learning
    for i in range(config.n_episodes):
        observation = env.reset()
        agent.init_history(observation)
        while True:
            pi = agent.get_policy(observation, i)
            if config.is_legal_action:
                action = np.random.choice(env.legal_action_list[observation], p = pi)
            else:
                action = np.random.choice(env.action_size, p = pi)
            next_observation, reward, done = env.step(action)
            agent.update(observation, action, next_observation, reward, done)
            if done:
                result.append(env.steps)
                np.save('{0}/result_{1}.npy'.format(save_dir, i_epoch), result)
                break
            else:
                observation = next_observation
