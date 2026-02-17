import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Question 1 – Generate & Plot Histograms
# -----------------------------------

def normal_histogram(n):
    # Generate n samples from Normal(0,1)
    data = np.random.normal(loc=0, scale=1, size=n)

    # Plot histogram
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Normal(0,1)")

    plt.show()

    return data


def uniform_histogram(n):
    # Generate n samples from Uniform(0,10)
    data = np.random.uniform(low=0, high=10, size=n)

    # Plot histogram
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Uniform(0,10)")

    plt.show()

    return data


def bernoulli_histogram(n):
    # Generate n samples from Bernoulli(0.5)
    data = np.random.binomial(n=1, p=0.5, size=n)

    # Plot histogram
    plt.hist(data, bins=10)
    plt.xlabel("Value (0 or 1)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Bernoulli(0.5)")

    plt.show()

    return data


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    return np.sum(data) / len(data)


def sample_variance(data):
    mean = sample_mean(data)
    n = len(data)

    return np.sum((data - mean) ** 2) / (n - 1)


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    sorted_data = np.sort(data)

    minimum = sorted_data[0]
    maximum = sorted_data[-1]
    median = np.median(sorted_data)

    q1 = np.percentile(sorted_data, 25, method="higher")
    q3 = np.percentile(sorted_data, 75, method="higher")

    return minimum, maximum, median, q1, q3


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    x = np.array(x)
    y = np.array(y)

    mean_x = sample_mean(x)
    mean_y = sample_mean(y)

    n = len(x)

    return np.sum((x - mean_x) * (y - mean_y)) / (n - 1)


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)

    return np.array([
        [var_x, cov_xy],
        [cov_xy, var_y]
    ])
