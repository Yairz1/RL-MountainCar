import gym
import numpy as np
from itertools import product
from numpy import array, exp, linspace
from random import random as rnd

from Actor_Critic.Model import Policy, V
from Actor_Critic.Utils import flush_print, simulate, padding_theta, argmax, create_matrix

ACTION_SPACE = [0, 1, 2]


def create_x(p, v):
    """
    Observation:
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    """

    c_p = linspace(-1.1, 0.4, 4)
    c_v = linspace(-0.06, 0.06, 8)
    c = product(c_p, c_v)
    return array([array([p, v]) - array([c1, c2]) for c1, c2 in c])


def create_phi(sigma_p, sigma_v):
    feature_map = {}

    def phi(p, v):
        if (p, v) not in feature_map:
            X = create_x(p, v)
            feature_map[p, v] = exp(-(X * X @ (1 / array([sigma_p, sigma_v]))) / 2)
        return feature_map[p, v]

    return phi


class Actor_Critic:

    def __init__(self, eps, v_alpha, policy_alpha, gamma, sigma_p, sigma_v, feature_space=32, plot_step=1e4):
        self.V = V(W=np.random.rand(feature_space))
        self.feature_space = feature_space
        self.policy = Policy(W=np.random.rand(feature_space * 3).reshape((feature_space * 3, 1)),
                             action_space=ACTION_SPACE)
        self.v_alpha = v_alpha
        self.policy_alpha = policy_alpha
        self.gamma = gamma
        self.plot_step = plot_step
        self.phi = create_phi(sigma_p, sigma_v)

    def actor_critic(self, env, max_total_steps=int(5e4)):
        steps = 0
        avg_list = []
        step_list = []
        avg_MC_list = []
        while steps <= max_total_steps:
            flush_print(f'\rCritic-Actor training process {100 * steps // max_total_steps}%')
            state = env.reset()
            I = 1
            for _ in range(500):
                # Collect information for the plot.
                self.collect_stats(avg_MC_list, avg_list, env, step_list, steps)

                steps += 1
                phi = self.phi(*state)
                mat = create_matrix(phi, ACTION_SPACE)
                action = self.policy(mat)
                new_state, R, done, info = env.step(action)
                next_phi = self.phi(*new_state)
                # delta :
                next_V = 0 if done else self.V(next_phi)
                delta = R + self.gamma * next_V - self.V(phi)
                # updating the weight vectors for the policy and the value functions
                self.V.update_parameters(dw=self.v_alpha * delta * phi)
                self.policy.update_parameters(
                    dw=self.policy_alpha * delta * I * self.policy.gradient(mat[:, action], action))
                I = self.gamma * I
                state = new_state
                if done: break

        flush_print(f'\rActor-Critic training process {100}%')
        return step_list, avg_list, avg_MC_list

    def collect_stats(self, avg_MC_list, avg_list, env, step_list, steps):
        if not steps % self.plot_step:
            avg_reward, avg_MC_reward = simulate(env, self.policy, self.gamma, self.phi, ACTION_SPACE,
                                                 episodes=100)
            avg_list.append(avg_reward)
            avg_MC_list.append(avg_MC_reward)
            step_list.append(steps)


def show_best_weights(gamma, sigma_p, sigma_v):
    env = gym.make('MountainCar-v0')
    W = np.load('Policy_weights.npy')
    policy = Policy(W, ACTION_SPACE)
    phi = create_phi(sigma_p, sigma_v)
    simulate(env, policy, gamma, phi, ACTION_SPACE)
    env.close()
