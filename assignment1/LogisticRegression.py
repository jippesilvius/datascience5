#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    """
    Logistic Regression class with optional L2 regularization.
    """

    def __init__(self, alpha=0.01, iter_n=1000, lambda_=0.0, scale=True, reg=True):
        self.alpha = alpha          # Learning rate
        self.iter_n = iter_n        # Number of iterations
        self.lambda_ = lambda_      # Regularization parameter
        self.scale = scale          # Whether to scale features
        self.reg = reg              # Enable regularization
        self.scaler = None          # Scaler object
        self.theta = None           # Parameter vector
        self.J_history = None       # Cost history

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        """
        Compute logistic regression cost with optional L2 regularization.
        """
        m = len(y)
        pred = self.sigmoid(X @ self.theta)  # shape (m,1)

        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-8
        term1 = -y * np.log(pred + epsilon)
        term2 = -(1 - y) * np.log(1 - pred + epsilon)
        cost = np.sum(term1 + term2) / m

        # Regularization (skip intercept)
        reg_term = 0
        if self.reg:
            reg_term = (self.lambda_ / (2 * m)) * np.sum(self.theta[1:] ** 2)
        return cost + reg_term

    def gradient_descent(self, X, y):
        """
        Perform gradient descent to optimize theta.
        """
        m = len(y)
        self.J_history = np.zeros(self.iter_n)

        for i in range(self.iter_n):
            pred = self.sigmoid(X @ self.theta)        # shape (m,1)
            error = pred - y                           # shape (m,1)
            grad = (X.T @ error) / m                   # shape (n,1)

            # Regularization term (skip theta[0])
            if self.reg:
                reg_term = np.zeros_like(self.theta)
                reg_term[1:] = (self.lambda_ / m) * self.theta[1:]
                grad += reg_term

            # Update parameters
            self.theta -= self.alpha * grad

            # Store cost
            self.J_history[i] = self.compute_cost(X, y)

    def fit(self, X, y):
        """
        Fit logistic regression model to X, y.
        """
        # Ensure y is a column vector
        y = y.reshape(-1, 1)

        # Scale features if needed
        if self.scale:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        # Add bias column
        X_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

        # Initialize theta
        self.theta = np.zeros((X_scaled.shape[1], 1))

        # Run gradient descent
        self.gradient_descent(X_scaled, y)
        return self

    def predict_proba(self, X):
        """
        Return probabilities for each sample.
        """
        if self.scale and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        X_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        return self.sigmoid(X_scaled @ self.theta)

    def predict(self, X, threshold=0.5):
        """
        Predict binary labels (0 or 1).
        """
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)

    def plot_cost(self):
        """
        Plot the cost function over iterations.
        """
        plt.plot(range(1, len(self.J_history) + 1), self.J_history, color='r')
        plt.xlabel('Iteration')
        plt.ylabel('Cost J')
        plt.title('Cost function over iterations')
        plt.show()
