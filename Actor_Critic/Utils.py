import sys
import numpy as np
import gym
import matplotlib.pyplot as plt
from numpy import zeros


def simulate(env, policy, gamma, phi, action_space, episodes=1):
    total_reward = 0.0
    monte_carlo_style_reward = 0
    for _ in range(episodes):
        observation = env.reset()

        for t in range(500):
            state = phi(*observation)
            if episodes == 1:
                env.render()

            mat = create_matrix(state, action_space)
            action = policy(mat)

            observation, reward, done, info = env.step(action)
            total_reward += reward
            monte_carlo_style_reward += reward * gamma ** t

            if done:
                if episodes == 1:
                    print(f'\nSuccess')
                break

    return total_reward / episodes, monte_carlo_style_reward / episodes


def padding_theta(theta, action):
    res = zeros(3 * theta.size)
    res[action * theta.size: (action + 1) * theta.size] = theta
    return res


def create_matrix(phi, action_space):
    rows = phi.size * len(action_space)
    cols = len(action_space)
    mat = np.zeros(shape=(rows, cols))
    for action in action_space:
        mat[:, action] = padding_theta(phi, action)
    return mat

def argmax(Q, theta, action_space):
    max = -float("inf")
    res = None
    Qs = [Q(padding_theta(theta, action)) for action in action_space]
    for i in range(len(Qs)):
        if Qs[i] > max:
            max = Qs[i]
            res = i
    return res


def flush_print(str):
    print(str, end="")
    sys.stdout.flush()


def show_and_save_plot(name, step_list, res, y_title):
    plt.plot(step_list, res)
    plt.xlabel("Steps")
    plt.ylabel(y_title)
    plt.title(name)
    plt.savefig(name + '.jpeg')
    plt.show()
