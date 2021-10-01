import numpy as np


class L1_Regularizer(object):
    ''' L1 Regularizer for Lasso Regression '''

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.sum(np.abs(w))

    def grad(self, w):
        return self.alpha * np.sign(w)


class L2_Regularizer(object):
    ''' L2 Regularizer for Ridge Regression '''

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.sum(w ** 2)

    def grad(self, w):
        return 2 * self.alpha * w