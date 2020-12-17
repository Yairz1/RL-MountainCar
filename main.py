import gym

from methods import SarsaAlgorithm

if __name__ == '__main__':
    eps = 0.25
    lambdas = [0.2, 0.7]
    alphas = [0.01, 0.1]
    gamma = 0.95
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 500
    algorithm = SarsaAlgorithm(eps, _lambda=0.5,
                               alpha=0.02,
                               gamma=1,
                               sigma_p=0.04,
                               sigma_v=0.0004,
                               feature_space=96,
                               plot_step=15e3)
    step_list, avg_reward, avg_MC = algorithm.sarsa(env)
    # show_and_save_plot(f'lambda: {_lambda} | alpha: {alpha}',step_list, avg_reward,"Average reward")
    # show_and_save_plot(f'(MC) lambda: {_lambda} | alpha: {alpha}',step_list, avg_MC,"Average MC reward")
    # simulate(env, algorithm.epsilon_greedy, gamma, episodes=1)
    env.close()
