import gym
import numpy as np
from itertools import product
from random import uniform
from numpy import diag, array, exp
from collections import defaultdict, Counter
from random import random as rnd
from Utils import flush_print, simulate, padding_theta, argmax


def create_x(p, v):
    """
    Observation:
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    """

    c_p = [uniform(a=-1.2, b=0.6) for _ in range(4)]
    c_v = [uniform(a=-0.07, b=0.07) for _ in range(8)]
    c = product(c_p, c_v)
    return array([array([p, v]) - array([c1, c2]) for c1, c2 in c])


def create_theta(sigma_p, sigma_v):
    feature_map = {}

    def theta(p, v):
        if (p, v) not in feature_map:
            X = create_x(p, v)
            feature_map[p, v] = exp(-(X * X @ (1 / array([sigma_p, sigma_v]))) / 2)
        return feature_map[p, v]

    return theta


class Q:
    def __init__(self, W):
        self.W = W

    def calc(self, theta):
        return theta.T @ self.W

    def __call__(self, theta):
        return self.calc(theta)

    def update_parameters(self, dw):
        self.W += dw


ACTION_SPACE = [0, 1, 2]


def create_epsilon_greedy_policy(Q, epsilon):
    def epsilon_greedy(theta, isGreedy=False):
        eps = 0 if isGreedy else epsilon
        best_greedy_action = argmax(Q, theta, ACTION_SPACE)
        rand_number = rnd()
        if rand_number <= eps / len(ACTION_SPACE) + 1 - eps:
            return best_greedy_action
        return np.random.choice(array([a for a in range(len(ACTION_SPACE)) if a != best_greedy_action]))

    return epsilon_greedy


class SarsaAlgorithm:

    def __init__(self, eps, _lambda, alpha, gamma, sigma_p, sigma_v, feature_space=96, plot_step=15e3):
        self.Q = Q(W=np.random.rand(feature_space))
        self.feature_space = feature_space
        self.epsilon_greedy = create_epsilon_greedy_policy(self.Q, eps)
        self._lambda = _lambda
        self.alpha = alpha
        self.gamma = gamma
        self.plot_step = plot_step
        self.theta = create_theta(sigma_p, sigma_v)

    def sarsa(self, env, max_total_steps=int(1e5)):
        steps = 0
        avg_list = []
        step_list = []
        avg_MC_list = []
        while steps <= max_total_steps:
            flush_print(f'\rSarsa training process {100 * steps // max_total_steps}%')
            state = env.reset()
            E = np.zeros(self.feature_space)
            theta = self.theta(*state)
            action = self.epsilon_greedy(theta)
            padded_theta = padding_theta(theta, action)
            for _ in range(200):
                steps += 1
                next_state, R, done, info = env.step(action)  # take a random action
                next_theta = self.theta(*next_state)
                new_action = self.epsilon_greedy(next_theta)
                next_padded_theta = padding_theta(next_theta, new_action)
                delta = R + self.gamma * self.Q(next_padded_theta) - self.Q(padded_theta)
                E = self._lambda * self.gamma * E + padded_theta
                self.Q.update_parameters(dw=self.alpha * delta * E)
                action = new_action
                padded_theta = next_padded_theta
                # Collect information for the plot.
                if not steps % self.plot_step:
                    avg_reward, avg_MC_reward = simulate(env, self.epsilon_greedy, self.gamma, self.theta, episodes=100)
                    avg_list.append(avg_reward)
                    avg_MC_list.append(avg_MC_reward)
                    step_list.append(steps)

                if done:
                    break

        flush_print(f'\rSarsa training process {100}%')
        return step_list, avg_list, avg_MC_list, self.theta
