#!/usr/bin/env python3
import numpy as np


class CustomGaussianNB:
    def __init__(self):
        self.labels = None
        self.prior_list = None
        self.mu = None
        self.sd = None

    def get_labels(self, y):
        """ Extract unique class labels from the training data """
        # YOUR CODE HERE

        return np.unique(y)

    def get_priors(self, y):
        """ Calculate prior probabilities for each unique label """
        prior_list = []
        y_len = len(y)
        labels = self.get_labels(y)
        ## For each unique label in labels:
        for label in labels:
            ##     Count the number of occurrences of the current label in labels
            count_labels = np.sum(y == label)
            ##     Divide the count by the total number of labels to get the prior probability
            prior_prob = count_labels / y_len
            ##     Append the prior probability to the priors list
            prior_list.append(prior_prob)
        ##     Return the list of priors
        ## YOUR SOLUTION HERE

        return prior_list, labels

    def get_likelihoods(self, X, y):
        self.prior_list, self.labels = self.get_priors(y)
        n_labels = len(self.labels)
        n_features = X.shape[1]
        #    Initialize mu as a 2D array of zeros with dimensions (number of labels, n)
        self.mu = np.zeros((n_labels, n_features))
        #    Initialize sd as a 2D array of zeros with dimensions (number of labels, n)
        self.sd = np.zeros((n_labels, n_features))

        #    For each label i from 0 to the number of unique labels:
        for i, label in enumerate(self.labels):
            #        Find the indices idx where y_train equals the current label
            idx = np.where(y == label)
            #        For each feature j from 0 to n:
            for j in range(n_features):
                #            Calculate the mean of the feature j for instances with label i and store it in mu[i][j]
                self.mu[i][j] = np.mean(X[idx, j])
                #            Calculate the standard deviation of the feature j for instances with label i and store it in sd[i][j]
                self.sd[i][j] = np.std(X[idx, j])
        return self


    def classify(self, X, priors, mu, sd):
        """
        Classify each test sample

        Args:
            - X_test: Test data samples, a matrix of dimensions (m, n)
            - labels: List of unique class labels
            - priors: List of prior probabilities for each class
            - mu: Matrix of class-wise means for each feature
            - sd: Matrix of class-wise standard deviations for each feature

        Returns:
            - y_pred: List of predicted class labels for each test sample
        """
        y_pred = []
        #  For each sample i
        for i in range(X.shape[0]):
            post = []
            #      For each class j:
            for j in range(len(self.labels)):
                #           Calculate squared differences: (X_test[i,:] - mu[j,:])^2
                sq_dif = (X[i, :] - mu[j, :]) ** 2
                #           Calculate norm_factors:  1 / sqrt(2 * pi * sd[j,:]
                norm = 1 / np.sqrt(2 * np.pi * sd[j, :] ** 2)
                #           Calculate likelihoods: norm_factors * exp(-squared_difference / (2 * sd[j,:])^2))
                like = norm * np.exp(-sq_dif / (2 * sd[j, :]))
                #           Append the likelihoods product multiplied by priors for class j to posteriors list
                post.append(np.prod(like) * priors[j])

            #      Get the highest posterior and use that as prediction for that sample
            y_pred.append(self.labels[np.argmax(post)])
        #  (See also the formula)
        return np.array(y_pred)

    def fit(self, X, y):
        self.get_likelihoods(X, y)


    def predict(self, X):
        self.classify(X, self.prior_list, self.mu, self.sd)