import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Models
from models.linear_models.RidgeRegression import RidgeRegression
from models.linear_models.LassoRegression import LassoRegression
from models.linear_models.LinearRegression import LinearRegression


def prep_data(path, test_size):
    """
    Prepares the data for the models.

    :param path: path to the data
    :param test_size: size of the test set
    :return: X_train, X_test, y_train, y_test
    """

    # Load the dataset
    df = pd.read_csv(path)
    
    # Preparing the data
    X = np.array(df.iloc[:, 0])
    y = np.array(df.iloc[:, 1])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':

    # Prepare data
    X_train, X_test, y_train, y_test = prep_data('data/scatter.csv', 0.3)

    # ------------------------------------------------------------ LinearRegression ------------------------------------------------------------    

    # Define model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # ------------------------------------------------------------ RidgeRegression ------------------------------------------------------------

    # Define model
    regressor = RidgeRegression(alpha = 0.01)
    regressor.fit(X_train, y_train, epochs = 1000, learning_rate = 0.0001)

    # ------------------------------------------------------------ LassoRegression ------------------------------------------------------------

    # Define model
    regressor = LassoRegression(alpha = 0.01)
    regressor.fit(X_train, y_train, epochs = 1000, learning_rate = 0.0001)






    # Make predictions
    y_pred = regressor.predict(X_test)

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