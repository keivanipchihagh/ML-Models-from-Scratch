import numpy as np
from Regression import Regression


class L1_Regularizer(object):
    ''' L1 Regularizer for Lasso Regression '''

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.sum(np.abs(w))

    def grad(self, w):
        return self.alpha * np.sign(w)


class LassoRegression(Regression):
    '''
    Lasso regression model
    '''

    def __init__(self, alpha: float) -> None:

        super().__init__()
        self.regularization = L1_Regularizer(alpha)
    

    def fit(self, X: np.array, y: np.array, epochs: int = 100, learning_rate: int = 0.0001) -> None:
        '''
        Fit the model to the training data X using Gradient Descent and L1 regularization

        :param X: training data
        :param y: target data
        :param epochs: number of epochs (Iterations) in case gradient decent is used (Default: 100)
        :param learning_rate: learning rate for gradient decent (Deafult: 0.0001)
        :return: None
        '''

        self.epochs = epochs
        self.learning_rate = learning_rate

        # Initialize weight
        self.initialize_weight(n_features = len(X))

        n_features = len(X)

        for _ in range(self.epochs):

            # Make prediction
            y_pred = np.dot(X, self.weight) + self.bias

            # Calculate error
            error = y - y_pred

            # Calculate gradient descent
            D_m = -(2 / n_features) * sum(X * (error))     # Partial Derivative for M (Weights)
            D_c = -(2 / n_features) * sum(error)           # Partial Derivative for C (Bias)

            # Apply L1 regularization (Lasso)
            D_m += self.regularization.grad(self.weight)

            # Update weights and bias
            self.weight -= (learning_rate * D_m)
            self.bias -= (learning_rate * D_c)