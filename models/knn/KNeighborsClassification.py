import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler

from utilities.distances import minkowski_distance


class KNeighborsClassification:

    def __init__(self, k: int = 3, p: int = 1) -> None:
        '''
        Initialize KNeighborsClassifier

        :params k: Number of neighbors to use for prediction
        :params p: Minkowski distance metric parameter
        :returns: None
        '''

        self.k = k
        self.p = p
    

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Fit the model for the spesific test point

        :params X: Training data
        :params y: Testing labels
        :returns neighbors: The k closest neighbors to the test point
        '''

        # Calculate distances
        distances = [minkowski_distance(x, y, self.p) for x in X]

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

            # Count the number of times each label appears in the k closest neighbors
            preds = Counter(y_train[neighbors.index])

            # Get the most common label and append it to y_pred list
            y_pred.append(preds.most_common()[0][0])
        
        return y_pred