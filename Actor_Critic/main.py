import gym
from Actor_Critic.Utils import show_and_save_plot, simulate
from Actor_Critic.methods import Actor_Critic, show_best_weights  # , show_best_weights

if __name__ == '__main__':
    ACTION_SPACE = [0, 1, 2]
    eps = 0.25
    v_alpha = 0.1
    policy_alpha = 0.1
    gamma = 1
    sigma_p = 0.04
    sigma_v = 0.0004
    show_best_weights(gamma, sigma_p, sigma_v)
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 500
    algorithm = Actor_Critic(eps,
                             v_alpha=v_alpha,
                             policy_alpha=policy_alpha,
                             gamma=gamma,
                             sigma_p=sigma_p,
                             sigma_v=sigma_v,
                             plot_step=5e3)
    step_list, avg_reward, avg_MC = algorithm.actor_critic(env)
    show_and_save_plot(f'v_alpha = {v_alpha} | policy_alpha={policy_alpha}', step_list, avg_MC,
                       "Average MC reward")
    simulate(env, algorithm.policy, gamma, algorithm.phi, ACTION_SPACE, episodes=1)
    # algorithm.policy.save()
    env.close()
