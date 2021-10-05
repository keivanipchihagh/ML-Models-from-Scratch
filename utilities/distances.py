def minkowski_distance(x, y, p = 1):    
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


def euclidean_distance(x, y):
    '''
    Calculates the Euclidean distance between two vectors x and y.

    :param x: x points
    :param y: y points
    :returns: the Euclidean distance between x and y
    :rasies: ValueError if x and y do not have the same length    
    '''

    return minkowski_distance(x, y, 2)


def manhattan_distance(x, y):
    '''
    Calculates the Manhattan distance between two vectors x and y.

    :param x: x points
    :param y: y points
    :returns: the Manhattan distance between x and y
    :rasies: ValueError if x and y do not have the same length    
    '''

    return minkowski_distance(x, y, 1)