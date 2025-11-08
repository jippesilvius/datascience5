#!/usr/bin/env python3

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold


class PipelineMaker:

    def __init__(self,  X, y, model_pipeline=None, model_param_grid=None, pipeline_grid_list=None):
        self.model_pipeline = model_pipeline
        self.model_param_grid = model_param_grid
        self.pipeline_grid_list = pipeline_grid_list
        self.X = X
        self.y = y
        self.cv_strategy = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.final_model = None


    def remove_low(self):
        rare_classes = self.y.value_counts()[self.y.value_counts() < 4].index  # less than 3 samples
        self.y = self.y.replace(rare_classes, 'Other')

    def get_cv_strategy(self):
        min_class_samples = self.y.value_counts().min()
        self.cv_strategy = StratifiedKFold(n_splits=min(min_class_samples, 5))

    def get_train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=20)

    def pipeline_acceptor(self):
        model_grid = GridSearchCV(self.model_pipeline, self.model_param_grid, cv=self.cv_strategy, scoring='accuracy')
        model_grid.fit(self.X_train, self.y_train)
        model_name = self.model_pipeline.steps[-1][0].upper()
        print(f"Best {model_name} params:", model_grid.best_params_)
        print(f"Best {model_name} score:", model_grid.best_score_)
        return model_grid.best_estimator_

    def execute_all(self):
        self.remove_low()
        self.get_cv_strategy()
        self.get_train_test_split()

        if self.pipeline_grid_list is not None:
            list_returns = []
            for pipeline, grid in self.pipeline_grid_list:
                self.model_pipeline = pipeline
                self.model_param_grid = grid
                list_returns.append(self.pipeline_acceptor())
            return list_returns

        else:
            return self.pipeline_acceptor()











