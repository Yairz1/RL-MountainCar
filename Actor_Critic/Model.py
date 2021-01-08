import numpy as np


class Model:
    def __init__(self, W, name):
        self.W = W
        self.name = name

    def save(self):
        np.save(self.name, self.W)


class V(Model):
    def __init__(self, W):
        super().__init__(W, "V_weights")
        self.W = W

    def calc(self, theta):
        return theta.T @ self.W

    def __call__(self, theta):
        return self.calc(theta)

    def update_parameters(self, dw):
        self.W += dw


class Policy(Model):
    def __init__(self, W, action_space):
        super().__init__(W, "Policy_weights")
        self.W = W
        self.save_for_gradient = None
        self.action_space = action_space

    def softmax(self, phi_matrix):
        '''
        :param phi_matrix: each column is the feature of the i'th action. e.g shape = 32x3
        :return:
        '''
        xs = self.W.T @ phi_matrix
        dist = np.exp(xs) / np.sum(np.exp(xs))
        dist = dist.reshape(-1)
        chosen_action = np.random.choice(self.action_space, p=dist)
        self.save_for_gradient = dist
        return chosen_action

    def __call__(self, phi):
        return self.softmax(phi)

    def gradient(self, phi, action):
        # todo write explanation in the document
        return phi * (1 - self.save_for_gradient[action])

    def update_parameters(self, dw):
        self.W += dw.reshape(-1, 1)
