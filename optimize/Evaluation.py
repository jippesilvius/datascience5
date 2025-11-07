#!/usr/env/bin python3

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix, roc_curve, roc_auc_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from numpy import mean



class Evaluator:
    def __init__(self, models, X, y):
        self.models = models
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models_tested = {}

    def remove_low_encounters(self):
        instances_x = len(self.X)
        instances_y = len(self.y)
        rare_class = self.y.value_counts()[self.y.value_counts() < 4].index
        self.X = self.X[~self.y.isin(rare_class)]
        self.y = self.y[~self.y.isin(rare_class)]
        print(f"Lost X: {instances_x-len(self.X)}/{instances_x} instances.")
        print(f"Lost X: {instances_y-len(self.y)}/{instances_y} instances.")

    def get_train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=20)
        return self


    def get_scores(self, model_name, model):
        model.fit(self.X_train, self.y_train)
        model_pred = model.predict(self.X_test)

        print(f"\n\n\nModel name: {model_name}")
        print(f"Precision: {precision_score(self.y_test, model_pred, average='weighted', zero_division=0)}")
        print(f"Accuracy: {accuracy_score(self.y_test, model_pred)}")
        print(f"Recall: {recall_score(self.y_test, model_pred, average='weighted', zero_division=0)}")
        print(f"F1 Score: {f1_score(self.y_test, model_pred, average='weighted', zero_division=0)}")


        print(f"ConfusionMatrix:\n Matrix: {confusion_matrix(self.y_test, model_pred)}")
        print(f"Classification report:\nReport {classification_report(self.y_test, model_pred, zero_division=0)}")

        if not model_name in self.models_tested:
            self.models_tested[model_name] = [model, model_pred]
        else:
            model_name = f"{model_name}_{len(self.models_tested)}"
            self.models_tested[model_name] = [model, model_pred]

        return model_name

    def plot_roc_curve(self, model_name, model):
        classes = np.unique(self.y_test)
        y_test_bin = label_binarize(self.y_test, classes=classes)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(self.X_test)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(self.X_test)
            if y_score.ndim == 1:
                y_score = np.vstack([1-y_score, y_score]).T
        else:
            fig = None
            self.models_tested[model_name].append(fig)
            return self.models_tested

        fig, ax = plt.subplots(figsize=(8,6))

        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            auc_score = roc_auc_score(y_test_bin[:, i], y_score[:, i])
            plt.plot(fpr, tpr, lw=2, label=f"Class {class_label}(AUC = {auc_score}" )

        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve: {model_name}")
        ax.legend(loc="lower right")
        ax.grid(True)

        self.models_tested[model_name].append(fig)
        plt.show()

        return self.models_tested


    def execute(self):
        self.remove_low_encounters()
        self.get_train_test_split()
        for model in self.models:
            model_name = model.__class__.__name__
            model_name = self.get_scores(model_name, model)
            self.plot_roc_curve(model_name, model)
        return self.models_tested