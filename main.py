import gym


if __name__ == '__main__':
    eps = 0.25
    lambdas = [0.2, 0.7]
    alphas = [0.01, 0.1]
    gamma = 0.95
    env = gym.make('MountainCar-v0')

    env.reset()
    for _ in range(10000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()
