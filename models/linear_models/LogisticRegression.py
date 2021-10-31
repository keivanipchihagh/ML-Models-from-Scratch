import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LogisticRegression:
    """
    Logistic Regression Classifier
    """

    def __init__(self, threshold = 0.5, iterations = 100):
        self.threshold = threshold
        self.iterations = iterations

    
    def sigmoid(self, X: np.array) -> np.array:
        """
        Sigmoid function

        :param X: Input
        :return: Output
        """
        return 1 / (1 + np.exp(-X))
    

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Train the model

        :param X: Input
        :param y: Output
        """
        self.X = X
        self.y = y
        self.theta = np.zeros(X.shape[1])
        self.loss = []

        for _ in range(self.iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.threshold * gradient

            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            self.loss.append(self.__loss(h, y))


    def predict_prob(self, X: np.array) -> np.array:
        """
        Predict the probability of the output

        :param X: Input
        :return: Output
        """
        return self.sigmoid(np.dot(X, self.theta))


    def predict(self, X: np.array) -> np.array:
        """
        Predict the output

        :param X: Input
        :param threshold: Threshold
        :return: Output
        """
        return self.predict_prob(X) >= self.threshold


    def __loss(self, h: np.array, y: np.array) -> float:
        """
        Calculate the loss

        :param h: Output
        :param y: Output
        :return: Loss
        """
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


if __name__ == '__main__':

    # Load the dataset
    df = pd.read_csv('data/iris.csv')
    
    # Preparing the data    
    # X = np.array(df.iloc[:, 0])
    # y = np.array(df.iloc[:, 1])
    X = df.drop(columns = 'target')
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # Define model
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = regressor.predict(X_test)
    print(y_pred)

    # Plotting the results
    # plt.figure(figsize = (5, 3))
    # plt.scatter(X_train, y_train , color = 'green')
    # plt.scatter(X_test, y_test , color = 'blue')
    # plt.plot(X_test , y_pred , color = 'r' , lw = 1)
    # plt.show()