#!/usr/bin/env python3

import numpy as np
import pandas as pd


class ID3classifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, y):
        # Calculate the entropy of a distribution for the classes in the target variable (y)."""
        # probs are the frequency of each class relative to the total number of labels (classes)
        # Entropy of the target (Vital Status) using sample.txt (no splitting) (using log2 as a base): 0.97
        y = np.array(y)
        classes = np.unique(y)
        y_zero_one = (y[:, np.newaxis] == classes).astype(int)

        probs = np.mean(y_zero_one, axis=0)
        entr = np.sum(-probs * np.log2(probs + 1e-20))
        return entr

    def information_gain(self, data, feature, target):
        # Calculate the information gain of partitioning the data by a specific feature
        # Information gain is the difference between the original entropy and the partition_entropy(subsets)
        # initial gain calculation feature Histological Type gain 0.00
        # initial gain calculation feature Grade gain 0.02
        # initial gain calculation feature Tumor Stage gain 0.07
        # initial gain calculation feature ER Status gain 0.02
        tot_entr = self.entropy(data[target].values)
        vals = data[feature].unique()
        weight_entr = 0
        for val in vals:
            sub_df = data[data[feature] == val]
            sub_prob = len(sub_df) / len(data)
            sub_ent = self.entropy(sub_df[target].values)
            weight_entr += sub_prob * sub_ent

        information_gains = tot_entr - weight_entr
        return information_gains

    def checks(self, features, data, target, depth):
        target_values = data[target].unique()
        if len(features) == 0:
            return data[target].mode()[0]
        if depth >= self.max_depth and self.max_depth != None:
            return data[target].mode()[0]
        if len(target_values) == 1:
            return target_values[0]

    def id3(self, data, features, target, depth=0):
        # ID3 algorithm to create a decision tree with max depth
        # Some checks before:
        check_result = self.checks(features, data, target, depth)
        if check_result is not None:
            return check_result

        # 1. select best feature based on information gain to split on (first split should be on Tumor Stage)
        # Stop if the best gain is 0 (no further information can be gained)
        gain_list = [self.information_gain(data, feature, target) for feature in features]

        if max(gain_list) == 0:
            return data[target].mode()[0]

        max_feature_indx = np.argmax(gain_list)
        max_feature_name = features[max_feature_indx]

        # 2. create a node for the selected feature
        tree = {max_feature_name: {}}

        # 3. split the data into subsets based on the possible values of the selected feature
        for value in data[max_feature_name].unique():
            sbuset = data[data[max_feature_name] == value]
            new_features = [f for f in features if f != max_feature_name]
            # 4. recursive apply algorithm on each subset, select next best feature until all samples in subset belong to a class or there are no more features
            leftover_subtree = self.id3(sbuset, new_features, target, depth + 1)
            # If no features are left, return the most common target value
            # Stop if the max depth is reached
            tree[max_feature_name][value] = leftover_subtree

        # 5. use majority voting for class if several class labels remain in a subset
        # If all data points have the same target value, return that value

        # Expected result to return:
        # {'Tumor Stage': {'stage iii': {'Histological Type': {'ilc': 'alive', 'other': 'dead', 'idc': 'alive'}},
        #                  'stage i': {'Histological Type': {'ilc': 'dead', 'mixed type': 'dead', 'other': 'alive', 'idc': 'alive'}},
        #                  'stage iv': {'Histological Type': {'ilc': 'alive', 'idc': 'dead'}},
        #                  'stage ii': {'Grade': {'grade 2': 'alive', 'grade 3': 'dead'}}
        #                 }
        # }
        return tree

    def predict_id3(self, tree, sample):

        if isinstance(tree, (str, int, np.integer, float)):
            return tree

        for feature, sub_tree in tree.items():
            sample_val = sample.loc[feature]

            if isinstance(sub_tree, dict) and sample_val in sub_tree:
                return self.predict_id3(sub_tree[sample_val], sample)

            elif isinstance(sub_tree, dict):
                leav_values = []

                for tree_branch in sub_tree.values():

                    if isinstance(tree_branch, (str, int, np.integer, float)):
                        leav_values.append(tree_branch)

                    elif isinstance(tree_branch, dict):
                        leav_values.append(self.predict_id3(tree_branch, sample))

                if leav_values:
                    uniques = set(leav_values)
                    return max(uniques, key=leav_values.count)

                else:
                    return "unknown"

            else:
                return sub_tree

    def fit(self, X, y):
        data = X.copy()
        data['target'] = y
        features = list(X.columns)
        self.tree = self.id3(data, features, 'target')
        return self.tree

    def predict(self, X):
        pred_list = []
        for _, row in X.iterrows():
            pred = self.predict_id3(self.tree, row)
            pred_list.append(pred)
        return pred_list
