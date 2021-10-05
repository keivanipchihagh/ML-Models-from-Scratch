import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Models
from models.knn.KNeighborsClassification import KNeighborsClassification
from models.knn.KNeighborsRegression import KNeighborsRegression


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
    X = df.drop(columns = 'target')
    y = df['target'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':


    # ------------------------------------------------------------ KNeighborsClassification ------------------------------------------------------------

    # Prepare data
    X_train, X_test, y_train, y_test = prep_data('data/iris.csv', 0.3)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)    

    # Define model
    knn = KNeighborsClassification(k = 5)

    # ------------------------------------------------------------ KNeighborsRegression ------------------------------------------------------------

    # Prepare data
    X_train, X_test, y_train, y_test = prep_data('data/boston.csv', 0.3)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)    

    # Define model
    knn = KNeighborsRegression(k = 5)





    # Make predictions
    y_pred = knn.predict(X_train, X_test, y_train)
    print(y_pred)