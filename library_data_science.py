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


def distance(X1: tuple[float], X2: tuple[float]) -> float:
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


def random_walk(start: tuple[float], weights: dict, steps = 100000, display = False) -> tuple[float]:
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


def derivative(f, x_0: float, delta_x = 0.0001) -> float:
    """Function returns the derivative of the given function at specified point.
    
       Function must be continuous at specified point."""
    
    return (f(x_0 + delta_x) - f(x_0)) / delta_x


def integral(f, a: float, b: float, delta_x = 0.0001) -> float:
    """Function returns the integral of the given function at specified range.
    
       Function must be continuous at specified range."""
    
    counter = 0

    while a <= b:
        counter = counter + (f(a) * delta_x)
        a = a + delta_x
    
    return counter


def area_monte_carlo(f, a: float, b: float, c: float, d: float, num_samples = 1000000) -> float:
    """Function returns the area of the given function at specified range.

       Method used in the process is Monte Carlo Simulation.

       Factors a, b are the range of x-axis, and factors c, d are the range of y-axis."""

    within_the_area = 0
    for sample in range(num_samples):
        x = random.uniform(a, b)
        y = random.uniform(c, d)

        try:
            if distance((x, 0), (x, y)) <= f(x):
                within_the_area = within_the_area + 1
        except ZeroDivisionError:
            pass

    estimated_area = (np.abs(a - b) * np.abs(c - d)) * (within_the_area / num_samples)

    return estimated_area
