import numpy as np


def gradient_decent(X: np.array, y: np.array, weight: float, bias: float, n_iterations: int, learning_rate: float, regularization = None) -> tuple:
    """
    Performs gradient descent on a given set of data points.

    :param X: Training data.
    :param y: Target data.
    :param weight: Weight parameter.
    :param bias: Bias parameter.
    :param n_iterations: Number of iterations.
    :param learning_rate: Learning rate.
    :return: Weight and bias parameters.
    """

    n_features = len(X)

    for _ in range(n_iterations):

        # Make prediction
        y_pred = np.dot(X, weight) + bias

        # Calculate error
        error = y - y_pred

        # Calculate gradient descent
        D_m = -(2 / n_features) * sum(X * (error))     # Partial Derivative for M (Weights)
        D_c = -(2 / n_features) * sum(error)           # Partial Derivative for C (Bias)

        # Apply regularization
        if regularization is not None:
            D_m += regularization.grad(weight)

        # Update weights and bias
        weight -= (learning_rate * D_m)
        bias -= (learning_rate * D_c)
    
    return weight, bias


def least_squares_fit(X: np.array, y: np.array) -> tuple:
    """
    Performs a least squares fit on a given set of data points.

    :param X: Training data.
    :param y: Target data.
    :return: Weight and bias parameters.
    """
    
    # Calculating means
    X_mean = np.mean(X)
    Y_mean = np.mean(y)

    num = 0
    den = 0

    # Calculating numerator and denominator
    for i in range(len(X)):
        num += (X[i] - X_mean) * (y[i] - Y_mean)
        den += (X[i] - X_mean) ** 2

    # Calculating weight
    weight = num / den

    # Calculating bias
    bias = Y_mean - weight * X_mean

    return weight, bias