import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class KNeighborsRegression:

    def __init__(self, k: int = 3, p: int = 1) -> None:
        '''
        Initialize KNeighborsRegressor

        :params k: Number of neighbors to use for prediction
        :params p: Minkowski distance metric parameter
        :returns: None
        '''

        self.k = k
        self.p = p
    

    def minkowski_distance(self, x, y, p = 1):
        '''
        Calculates the Minkowski distance between two vectors x and y.

        :param x: x points
        :param y: y points
        :param p: p is the Minkowski power parameter. (Default = 1)
        :returns: the Minkowski distance between x and y
        :rasies: ValueError if x and y do not have the same length    
        '''

        # Exception handling
        if len(x) != len(y):
            raise ValueError('x and y must be of same length')

        n = len(x)

        distance  = 0
        for i in range(n):
            distance  += (abs(x[i] - y[i]) ** p)

        return distance ** (1 / p)
    

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Fit the model for the spesific test point

        :params X: Training data
        :params y: Testing labels
        :returns neighbors: The k closest neighbors to the test point
        '''

        # Calculate distances
        distances = [self.minkowski_distance(x, y, self.p) for x in X]

        # Return neighbors
        return pd.DataFrame(
            data = distances,
            columns = ['dist']
        ).sort_values(
            by = 'dist',
            axis = 0
        )[:self.k]


    def predict(self, X_train: np.array, X_test: np.array, y_train: np.array) -> np.array:
        '''
        Predict the labels for the test data

        :params X_train: Training data
        :params X_test: Testing data
        :params y_train: Training labels
        :returns y_pred: Predicted labels
        '''
        
        y_pred = []
        for y in X_test:

            # Fit the model for the spesific test point
            neighbors = self.fit(X_train, y)

            # Calcualte mean of the neighbors append it to y_pred list
            y_pred.append(y_train[neighbors.index].mean())
        
        return y_pred




if __name__ == '__main__':

    # Load the dataset
    df = pd.read_csv('data/boston.csv')
    
    # Preparing the data
    X = df.drop(columns = 'target')
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)    

    # Define model
    knn = KNeighborsRegression(k = 5)

    # Make predictions
    y_pred = knn.predict(X_train, X_test, y_train)
    print(y_pred)