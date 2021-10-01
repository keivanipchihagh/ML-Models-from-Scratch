import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from costs import MSE


class Regression:
    '''
    Base regression model
    '''

    def __init__(self, n_iterations: int = 100, learning_rate: int = 0.001) -> None:
        '''
        Initialize the model hyper parameters

        :param iterations: number of iterations
        :param learning_rate: learning rate
        '''

        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.n_features = None

        # Parameters
        self.bias = 0
        self.weights = []

        # Metrics
        self.training_mse = []


    def initialize_weights(self, n_features: int) -> np.ndarray:
        '''
        Initialize the weights of the model

        :param n_features: number of features
        :return: random weights between 0 and 1
        '''

        self.weights = np.random.rand()


    def fit(self, X: np.array, y: np.array) -> None:
        '''
        Fit the model to the training data X using Gradient Descent

        :param X: input data
        :param y: target data
        :return: None
        '''

        self.n_features = len(X)

        # Initialize weights
        self.initialize_weights(n_features = self.n_features)

        # Gradient descent optimizaion
        for _ in range(self.n_iterations):

            # Make prediction on X
            y_pred = self.predict(X)

            # Calculate error
            error = y - y_pred

            # Calculate cost
            mse = MSE(y, y_pred)
            self.training_mse.append(mse)

            # Calculate gradient descent
            D_m = -(2 / self.n_features) * sum(X * (error))     # Partial Derivative for M (Weights)
            D_c = -(2 / self.n_features) * sum(error)           # Partial Derivative for C (Bias)

            # Update weights and bias
            self.weights -= (self.learning_rate * D_m)
            self.bias -= (self.learning_rate * D_c)
    

    def predict(self, X: np.array) -> np.array:
        '''
        Predict the output of the model

        :param X: input data
        :return: predicted output
        '''

        return X.dot(self.weights) + self.bias




if __name__ == '__main__':

    # Load the dataset
    df = pd.read_csv('data/data.csv')
    
    # Preparing the data
    X = np.array(df.iloc[:, 0])
    y = np.array(df.iloc[:, 1])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # Initialize the model
    regressor = Regression(n_iterations = 1000, learning_rate = 0.0001)

    # Training the model
    regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = regressor.predict(X_test)

    print(regressor.weights, regressor.bias)

    # Plotting the results
    plt.figure(figsize = (5, 3))
    plt.scatter(X_train, y_train , color = 'green')
    plt.scatter(X_test, y_test , color = 'blue')
    plt.plot(X_test , y_pred , color = 'r' , lw = 1)
    plt.show()





# class Regression:
#     """
#     Regression class: Performs Linear Regression
#     """
#     def __init__(self, data, targets, test_data, test_targets, batch_size,
#                  epochs, learning_rate, verbose=False):
#         """
#         Initialize Regression class
#         """
#         self.data = data
#         self.targets = targets
#         self.test_data = test_data
#         self.test_targets = test_targets
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.learning_rate = learning_rate
#         self.verbose = verbose
#         self.weights = None
#         self.bias = None
#         self.costs = []
#         self.final_cost = None
#         self.final_weights = None
#         self.final_bias = None
#         self.final_predictions = None

#     def fit(self):
#         """
#         Fit the model to the data
#         """
#         self.weights = np.random.rand(self.data.shape[1], 1)
#         self.bias = np.random.rand(1)
#         for epoch in range(self.epochs):
#             for i in range(0, self.data.shape[0], self.batch_size):
#                 x = self.data[i:i + self.batch_size]
#                 y = self.targets[i:i + self.batch_size]
#                 predictions = np.dot(x, self.weights) + self.bias
#                 cost = self.cost(predictions, y)
#                 self.costs.append(cost)
#                 self.weights -= self.learning_rate * np.dot(x.T,
#                                                              (predictions - y))
#                 self.bias -= self.learning_rate * (predictions - y).sum()
#             if self.verbose:
#                 print("Epoch: {} - Cost: {}".format(epoch, cost))
#         self.final_cost = cost
#         self.final_weights = self.weights
#         self.final_bias = self
    
#     def cost(self, predictions, y):
#         """
#         Calculate the cost of the model
#         """
#         return np.mean((predictions - y) ** 2)