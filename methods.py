import numpy as np
from itertools import product
from random import uniform
from numpy import diag, array, exp


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
    return array([array([p, v]) - array([c1, c2]) for c1, c2 in c]).T


def create_theta(sigma_p, siga_v):
    def theta(p, v):
        X = create_x(p, v)
        return exp(-(X.T @ (1 / diag([sigma_p, siga_v])) @ X) / 2)

    return theta


class Q:
    def __init__(self, W):
        self.W = W

    def calc(self, p, v, theta):
        return theta.T @ self.W


create_x(0.2, 0.05)
