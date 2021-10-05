import numpy as np


class Regression:
    '''
    Base regression model
    '''

    def __init__(self) -> None:

        # Parameters
        self.bias = 0
        self.weight = []
        self.regularization = None

        # Metrics
        self.training_mse = []


    def initialize_weight(self, n_features: int) -> np.ndarray:
        '''
        Initialize the weight of the model randomly

        :param n_features: number of features
        :return: random weight between 0 and 1
        '''

        self.weight = np.random.rand()
    

    def predict(self, X: np.array) -> np.array:
        '''
        Predict the output of the model in form of array

        :param X: input data
        :return: predicted output numpy array
        '''

        return X.dot(self.weight) + self.bias