import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Regression import Regression
from sklearn.model_selection import train_test_split


class LinearRegression(Regression):
    '''
    Linear regression model
    '''

    def __init__(self) -> None:
        super().__init__()


    def fit(self, X: np.array, y: np.array) -> None:
        '''
        Fit the model weight and bias to the training data X using batch optimization by Least Squares

        :param X: training data
        :param y: target data
        :return: None
        '''

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
        self.weight = num / den

        # Calculating bias
        self.bias = Y_mean - self.weight * X_mean



if __name__ == '__main__':

    # Load the dataset
    df = pd.read_csv('data/scatter.csv')
    
    # Preparing the data    
    X = np.array(df.iloc[:, 0])
    y = np.array(df.iloc[:, 1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # Define model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = regressor.predict(X_test)

    # Plotting the results
    plt.figure(figsize = (5, 3))
    plt.scatter(X_train, y_train , color = 'green')
    plt.scatter(X_test, y_test , color = 'blue')
    plt.plot(X_test , y_pred , color = 'r' , lw = 1)
    plt.show()