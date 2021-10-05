import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Regression import Regression
from sklearn.model_selection import train_test_split


class L2_Regularizer(object):
    ''' L2 Regularizer for Ridge Regression '''

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.sum(w ** 2)

    def grad(self, w):
        return 2 * self.alpha * w


class RidgeRegression(Regression):
    '''
    Ridge regression model (Inherited from Regression class)
    - L2 Regularizartion used
    '''

    def __init__(self, alpha: float) -> None:
        '''
        Initialize the model

        :param alpha: L2 regularization parameter
        '''
        
        super().__init__()
        self.regularization = L2_Regularizer(alpha)
    

    def fit(self, X: np.array, y: np.array, epochs: int = 100, learning_rate: int = 0.0001) -> None:
        '''
        Fit the model to the training data X using Gradient Descent and L2 regularization

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

            # Apply L2 regularization (Ridge)
            D_m += self.regularization.grad(self.weight)

            # Update weights and bias
            self.weight -= (learning_rate * D_m)
            self.bias -= (learning_rate * D_c)



if __name__ == '__main__':

    # Load the dataset
    df = pd.read_csv('data/scatter.csv')
    
    # Preparing the data    
    X = np.array(df.iloc[:, 0])
    y = np.array(df.iloc[:, 1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # Define model
    regressor = RidgeRegression(alpha = 0.01)
    regressor.fit(X_train, y_train, epochs = 1000, learning_rate = 0.0001)

    # Make predictions
    y_pred = regressor.predict(X_test)

    # Plotting the results
    plt.figure(figsize = (5, 3))
    plt.scatter(X_train, y_train , color = 'green')
    plt.scatter(X_test, y_test , color = 'blue')
    plt.plot(X_test , y_pred , color = 'r' , lw = 1)
    plt.show()