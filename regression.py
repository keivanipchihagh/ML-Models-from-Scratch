import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from costs import MSE
from optimizers import gradient_decent, least_squares_fit
from regularizers import L1_Regularizer, L2_Regularizer


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
        self.weight = []

        # Metrics
        self.training_mse = []


    def initialize_weight(self, n_features: int) -> np.ndarray:
        '''
        Initialize the weight of the model

        :param n_features: number of features
        :return: random weight between 0 and 1
        '''

        self.weight = np.random.rand()


    def fit(self, X: np.array, y: np.array, use_gradient_decent: bool = True) -> None:
        '''
        Fit the model to the training data X using Gradient Descent

        :param X: input data
        :param y: target data
        :return: None
        '''

        self.n_features = len(X)

        # Initialize weight
        self.initialize_weight(n_features = self.n_features)
        
        if use_gradient_decent:
            # Gradient descent optimizaion
            self.weight, self.bias = gradient_decent(
                X = X,
                y = y,
                weight = self.weight,
                bias = self.bias,
                n_iterations = self.n_iterations,
                learning_rate = self.learning_rate,
            )
        else:
            # Least squares optimizaion
            self.weight, self.bias = least_squares_fit(
                X = X,
                y = y,
            )
    

    def predict(self, X: np.array) -> np.array:
        '''
        Predict the output of the model

        :param X: input data
        :return: predicted output
        '''

        return X.dot(self.weight) + self.bias




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
    regressor.fit(X_train, y_train, use_gradient_decent = False)

    # Make predictions
    y_pred = regressor.predict(X_test)

    print(regressor.weight, regressor.bias)

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
#         self.weight = None
#         self.bias = None
#         self.costs = []
#         self.final_cost = None
#         self.final_weight = None
#         self.final_bias = None
#         self.final_predictions = None

#     def fit(self):
#         """
#         Fit the model to the data
#         """
#         self.weight = np.random.rand(self.data.shape[1], 1)
#         self.bias = np.random.rand(1)
#         for epoch in range(self.epochs):
#             for i in range(0, self.data.shape[0], self.batch_size):
#                 x = self.data[i:i + self.batch_size]
#                 y = self.targets[i:i + self.batch_size]
#                 predictions = np.dot(x, self.weight) + self.bias
#                 cost = self.cost(predictions, y)
#                 self.costs.append(cost)
#                 self.weight -= self.learning_rate * np.dot(x.T,
#                                                              (predictions - y))
#                 self.bias -= self.learning_rate * (predictions - y).sum()
#             if self.verbose:
#                 print("Epoch: {} - Cost: {}".format(epoch, cost))
#         self.final_cost = cost
#         self.final_weight = self.weight
#         self.final_bias = self
    
#     def cost(self, predictions, y):
#         """
#         Calculate the cost of the model
#         """
#         return np.mean((predictions - y) ** 2)