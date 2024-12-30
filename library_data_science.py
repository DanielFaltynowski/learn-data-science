import random
import numpy as np
import matplotlib.pyplot as plt



def mean(data: list) -> float:
    """Calculates the mean of the dataset containing numbers (integers or/and floats).
       
       Mean represents the central value of the dataset."""
    
    counter = 0

    for datum in data:
        counter = counter + datum

    return counter /len(data)


def variance(data: list) -> float:
    """Calculates the variance of the dataset containing numbers (integers or/and floats).
       
       Variance indicates how much the data values differ from the mean. The bigger variance, the more spread out the data are."""
    
    counter = 0
    data_mean = mean(data)

    for datum in data:
        counter = counter + (datum - data_mean) ** 2

    return counter /len(data)


def standard_deviation(data: list) -> float:
    """Calculates the standard deviation of the dataset containing numbers (integers or/and floats).
    
       Standard Deviation provides an easy-to-interpret measure of dispersion expressed in the same unit as the data."""
    
    return variance(data) ** 0.5


def std(data: list) -> float:
    """Calculates the standard deviation of the dataset containing numbers (integers or/and floats).
    
       Standard Deviation provides an easy-to-interpret measure of dispersion expressed in the same unit as the data."""
    
    return standard_deviation(data)


def distance(X1: tuple, X2: tuple) -> float:
    """Calculates the distance between points X1 and X2 on euclidean space.
    
       Sizes of X1 and X2 should be equal."""
    
    if not (len(X1) == len(X2)):
        print('Invalid input!')
        return -1
    else:
        counter = 0
        for i in range(len(X1)):
            counter = counter + ((X1[i] - X2[i]) ** 2)
        return counter ** 0.5


def random_walk(start: tuple[float, float], weights: dict, steps = 100000, display = False) -> tuple[float, float]:
    """Function simulates the random walk algorithm.

       Dictionary of weights should contain number of steps in every side {left, right, up, down}.
        
       It is highly recomended to use this function in another function to clearly defined purpose.
       
       If the input data will be wrong, function returns -1."""

    if 'left' not in weights or 'right' not in weights or 'up' not in weights or 'down' not in weights:
        print('Invalid input!')
        return -1
    else:
        # Storage of points
        X = []
        Y = []

        # Actual position
        x = start[0]
        y = start[1]

        # Walk
        for attempt in range(steps):
            horizontal_vertical = random.choice(['left', 'right', 'up', 'down'])
            
            match horizontal_vertical:
                case 'left':
                    x = x - weights['left']
                case 'right':
                    x = x + weights['right']
                case 'up':
                    y = y + weights['up']
                case 'down':
                    y = y - weights['down']

            X.append(x)
            Y.append(y)
        
        # Diagram using matplotlib
        if display:
            plt.plot(X, Y, zorder=1)
            plt.scatter(start[0], start[1], color='orange', label='start', linewidths=5, zorder=2)
            plt.scatter(x, y, color='red', label='end', linewidths=5, zorder=2)
            plt.legend()
            plt.show()
        
        # Return the actual position
        return (x, y)
