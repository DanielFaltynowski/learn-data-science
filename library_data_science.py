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


def var_p(data: list) -> float:
    """Calculates the variance of the population containing numbers (integers or/and floats).
       
       Variance indicates how much the data values differ from the mean. The bigger variance, the more spread out the data are."""
    
    counter = 0
    data_mean = mean(data)

    for datum in data:
        counter = counter + (datum - data_mean) ** 2

    return counter / len(data)


def var_s(data: list) -> float:
    """Calculates the variance of the sample containing numbers (integers or/and floats).
       
       Variance indicates how much the data values differ from the mean. The bigger variance, the more spread out the data are."""
    
    counter = 0
    data_mean = mean(data)

    for datum in data:
        counter = counter + (datum - data_mean) ** 2

    return counter / (len(data) - 1)


def std_p(data: list) -> float:
    """Calculates the standard deviation of the population containing numbers (integers or/and floats).
    
       Standard Deviation provides an easy-to-interpret measure of dispersion expressed in the same unit as the data."""
    
    return var_p(data) ** 0.5


def std_s(data: list) -> float:
    """Calculates the standard deviation of the sample containing numbers (integers or/and floats).
    
       Standard Deviation provides an easy-to-interpret measure of dispersion expressed in the same unit as the data."""
    
    return var_s(data) ** 0.5


def standard_error(sample: list):
    """Calculates the standard error of the dataset containing numbers (integers or/and floats).
       
       Standard error measures the uncertainty of the mean estimate, which decreases as the sample size increases."""

    return std_s(sample) / (len(sample) ** 0.5)


def se(sample: list):
    """Calculates the standard error of the well-selected sample from the population (dataset) containing numbers (integers or/and floats).
       
       Standard error measures the uncertainty of the mean estimate, which decreases as the sample size increases."""
    
    return standard_error(sample)


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


def area_monte_carlo(f, a: float, b: float, c: float, d: float, num_points = 1000000) -> float:
    """Function returns the area of the given function at specified range.

       Method used in the process is Monte Carlo Simulation.

       Factors a, b are the range of x-axis, and factors c, d are the range of y-axis."""

    within_the_area = 0
    for sample in range(num_points):
        x = random.uniform(a, b)
        y = random.uniform(c, d)

        try:
            if distance((x, 0), (x, y)) <= f(x):
                within_the_area = within_the_area + 1
        except ZeroDivisionError:
            pass

    estimated_area = (np.abs(a - b) * np.abs(c - d)) * (within_the_area / num_points)

    return estimated_area


def pdf_uniform(x: float, a: float, b: float) -> float:
    """The function generates the probability density function values of a uniform distribution."""

    if a <= x <= b:
        return 1 / (b - a)
    else: 
        return 0


def pdf_normal(x: float, mean: float, std: float) -> float:
    """The function generates the probability density function values of a normal distribution."""

    return ( 2.718281828459045 ** (- ((x - mean) ** 2) / (2 * (std ** 2)))) / ( std * ((2 * 3.141592653589793) ** 0.5) )


def pdf_exponential(x: float, l: float) -> float:
    """The function generates the probability density function values of a exponential distribution."""

    return l * (2.718281828459045 ** (-l * x))


def polynomial(x: float, factors: tuple) -> float:
    """Function returns the value for the given polynomial.
    
       Variable `factors` should contain the factors of the polynomial.
    
       For example for 3x^3 - 8x - 1, factors = (3, 0, -8, -1).

       For example for 0.5x^2 + 3x - 5, factors = (0.5, 3, -5)."""

    counter = 0

    for i in range(len(factors)):
        counter = counter + ( factors[i] * (x ** (len(factors) - i - 1)) )
    
    return counter


def residual_sum_of_squares(Y_observed: list, Y_predicted: list) -> float:
    """Calculates the residual sum of squared of the fit for the two-dimentional dataset containing pairs of numbers (integers or/and floats).
       
       Represents the total error for the fit. In other words, `RSS` repserents the sum of the distances between the observed
       and predicted values."""

    if not len(Y_observed) == len(Y_predicted):
        print('Invalid input!')
    else:
        counter = 0
        
        for i in range(len(Y_observed)):
            counter = counter + (Y_observed[i] - Y_predicted[i]) ** 2
        
        return counter


def rss(Y_observed: list, Y_predicted: list) -> float:
    """Calculates the residual sum of squared of the fit for the two-dimentional dataset containing pairs of numbers (integers or/and floats).
       
       Represents the total error for the fit. In other words, `RSS` repserents the sum of the distances between the observed
       and predicted values."""
    
    return residual_sum_of_squares(Y_observed, Y_predicted)


def mean_square_error(Y_observed: list, Y_predicted: list) -> float:
    """Calculates the mean square of the two-dimentional dataset containing pairs of numbers (integers or/and floats).
    
       Represents average error for the fit. `MSE` is useful for comparing two different models for the same data. The smaller `MSE`, 
       the better curve is fitted.
       
       But for large errors this measurement can be extremally bigger than it really matters for us."""

    if not len(Y_observed) == len(Y_predicted):
        print('Invalid input!')
    else:
        return residual_sum_of_squares(Y_observed, Y_predicted) / len(Y_observed)


def mse(Y_observed: list, Y_predicted: list) -> float:
    """Calculates the mean square of the two-dimentional dataset containing pairs of numbers (integers or/and floats).
    
       Represents average error for the fit. `MSE` is useful for comparing two different models for the same data. The smaller `MSE`, 
       the better curve is fitted.
       
       But for large errors this measurement can be extremally bigger than it really matters for us."""

    return mean_square_error(Y_observed, Y_predicted)


def coefficient_of_determination(Y_observed: list, Y_predicted: list) -> float:
    """Calculates the coefficient of determination of the fit for the two-dimentional dataset containing pairs of numbers 
       (integers or/and floats).
    
       `R^2`: Shows how good is the fit. `R^2` is intended to capture the proportion of variability in a dataset that is 
       accounted for by the statistical model provided by the fit.
       
       If `R^2 = 1`, then we got perfect fit. If `R^2 = 0`, then no data changes are explained."""

    if not len(Y_observed) == len(Y_predicted):
        print('Invalid input!')
    else:
        return 1 - (residual_sum_of_squares(Y_observed, Y_predicted) / (variance(Y_observed) * len(Y_observed)))


def r_square(Y_observed: list, Y_predicted: list) -> float:
    """Calculates the coefficient of determination of the fit for the two-dimentional dataset containing pairs of numbers 
       (integers or/and floats).
    
       `R^2`: Shows how good is the fit. `R^2` is intended to capture the proportion of variability in a dataset that is 
       accounted for by the statistical model provided by the fit.
       
       If `R^2 = 1`, then we got perfect fit. If `R^2 = 0`, then no data changes are explained."""

    return coefficient_of_determination(Y_observed, Y_predicted)
