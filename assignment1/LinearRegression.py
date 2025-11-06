#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np


class LinearRegression:
    """
    Linear regression class to execute linear regression.
    """

    def __init__(self, alpha, lambda_, iter_n=100, scale=True, reg=True):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.iter_n = iter_n
        self.scale = scale
        self.scaler = None
        self.theta = None
        self.J_history = np.zeros(iter_n)
        self.reg = reg

    def compute_cost(self, X, y):
        # 1.Determine the number of data points
        m = len(X)
        # 2.Determine the prediction
        pred = X @ self.theta
        # 3.Calculate the difference between this prediction and the actual value
        err = pred - y

        # 4.square this difference
        sq_err = err.T @ err
        # 5.Add all these squares together and divide by twice the number of data points
        cost = sq_err / (2 * m)
        reg = (self.lambda_ / (2 * m)) * np.sum(self.theta[1:] ** 2)
        J = cost + reg

        return J

    def gradient_descent(self, X, y):
        m = len(y)

        for i in range(self.iter_n):
            # Determine the prediction for the data point, given the current value of theta
            pred = X @ self.theta
            # Determine the difference between this forecast and the true value
            err = pred - y

            # Regularization step (theta 0 not regularized)
            reg = (self.lambda_ / m) * self.theta
            reg[0] = 0

            # Multiply this difference by the ith value of X + reg.
            gradient = (1 / m) * X.T @ err + reg

            # Update the ith parameter of theta, namely by decreasing it by
            self.theta = self.theta - self.alpha * gradient
            self.J_history[i] = self.compute_cost(X, y)

        return self.theta, self.J_history

    def fit(self, X, y):
        if self.scale:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        X_scaled = np.c_[np.ones((X.shape[0], 1)), X_scaled]
        self.theta = np.zeros(X_scaled.shape[1])
        self.gradient_descent(X_scaled, y)
        return self

    def predict(self, X):
        if self.scale and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        X_scaled = np.c_[np.ones((X.shape[0], 1)), X_scaled]
        return X_scaled @ self.theta

    def reverse_theta(self):
        """
        Transforms the theta parameters obtained from a regression model on scaled data
        back to the original (non-scaled) feature space.

        Parameters:
        theta (numpy array): Array of theta parameters obtained from the regression model.
        scaler (StandardScaler object): The scaler object used to scale the features.

        Returns:
        numpy array: transformed theta parameters in the context of the original, non-scaled features.

        The function performs the following steps:
        1. Initializes a new array `theta_original` with the same shape as `theta`.
        2. Corrects the intercept (theta[0]) to account for the means of the original features.
        3. Adjusts the coefficients (theta[1:]) by reversing the effect of scaling using the standard deviations.
        """

        if not self.scale or self.scaler is None:
            return self.theta
        # initialize
        theta_original = np.zeros_like(self.theta)
        # tranform back intercept
        theta_original[0] = self.theta[0] - np.sum((self.theta[1:] * self.scaler.mean_) / self.scaler.scale_)
        # transform back coefficients
        theta_original[1:] = self.theta[1:] / self.scaler.scale_
        return theta_original

    def plot_cost(self):
        plt.plot(range(1, len(self.J_history) + 1), self.J_history, color='r')
        plt.xlabel('iterations')
        plt.ylabel('cost J')
        plt.title('cost over iterations')
        plt.show()