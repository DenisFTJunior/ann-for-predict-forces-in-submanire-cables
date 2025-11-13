"""Regression models following a common pattern similar to the provided LogisticRegressionModel.

Includes:
- LinearRegressionModel
- KNNRegressionModel
- RandomForestRegressionModel
- SVRRegressionModel
- XGBoostRegressionModel

All classes:
- Initialize with data; internally use DataProcessor + DataTransformer('excel') to split.
- train(): fit a baseline model on train split.
- train_with_grid(): run GridSearchCV over param_grid (cv=3) using neg MSE scoring.
- predict(): predict with baseline model on test split.
- predict_with_grid(): predict with best grid model on test split.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from data.DataProcessor import DataProcessor
from data.DataTransformer import DataTransformer

class _BaseRegressor:
    x_train = x_test = y_train = y_test = x_val = y_val = None

    def __init__(self, data):
        data_processor = DataProcessor(data, DataTransformer.get_transformer('excel'), '% de carga nas ancoras')
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = data_processor.split_data()

    # Utility to build a GridSearchCV consistently for regressors
    def _grid(self, model, param_grid):
        return GridSearchCV(
            model,
            param_grid=param_grid,
            cv=3,
            verbose=True,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
        )


class LinearRegressionModel(_BaseRegressor):
    LIN = None
    LIN_best = None
    params_best: Optional[dict] = None
    param_grid = [{
        'fit_intercept': [True, False],
        'copy_X': [True],
        'positive': [False, True],
    }]

    def train(self):
        self.LIN = LinearRegression().fit(self.x_train, self.y_train)
        return self.LIN

    def train_with_grid(self):
        model = LinearRegression()
        clf = self._grid(model, self.param_grid)
        self.LIN_best = clf.fit(self.x_train, self.y_train)
        self.params_best = self.LIN_best.best_params_
        return self.LIN_best, self.params_best

    def predict(self):
        return self.LIN.predict(self.x_test)

    def predict_with_grid(self):
        return self.LIN_best.predict(self.x_test)


class SVRRegressionModel(_BaseRegressor):
    SVR_ = None
    SVR_best = None
    params_best: Optional[dict] = None
    param_grid = [{
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'C': list(np.logspace(-3, 3, 7)),
        'epsilon': [0.1, 0.2, 0.5, 1.0],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4],
    }]

    def train(self):
        self.SVR_ = SVR(kernel='rbf', C=1.0, epsilon=0.1).fit(self.x_train, self.y_train)
        return self.SVR_

    def train_with_grid(self):
        model = SVR()
        clf = self._grid(model, self.param_grid)
        self.SVR_best = clf.fit(self.x_train, self.y_train)
        self.params_best = self.SVR_best.best_params_
        return self.SVR_best, self.params_best

    def predict(self):
        return self.SVR_.predict(self.x_test)

    def predict_with_grid(self):
        return self.SVR_best.predict(self.x_test)



class RandomForestRegressionModel(_BaseRegressor):
    RF = None
    RF_best = None
    params_best: Optional[dict] = None
    param_grid = [{
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'max_features': ['auto', 'sqrt', 'log2']
    }]

    def train(self):
        self.RF = RandomForestRegressor(random_state=4, n_estimators=200).fit(self.x_train, self.y_train)
        return self.RF

    def train_with_grid(self):
        model = RandomForestRegressor(random_state=4)
        clf = self._grid(model, self.param_grid)
        self.RF_best = clf.fit(self.x_train, self.y_train)
        self.params_best = self.RF_best.best_params_
        return self.RF_best, self.params_best

    def predict(self):
        return self.RF.predict(self.x_test)

    def predict_with_grid(self):
        return self.RF_best.predict(self.x_test)
    

class XGBoostRegressionModel(_BaseRegressor):
    XGB = None
    XGB_best = None
    params_best: Optional[dict] = None
    # Keep modest search space to avoid huge runs by default
    param_grid = [{
        'n_estimators': [200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_lambda': [1.0, 10.0],
        'reg_alpha': [0.0, 1.0],
    }]


    def train(self):
        self.XGB = XGBRegressor(
            random_state=4,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=1.0,
            colsample_bytree=1.0,
            tree_method='hist',
        ).fit(self.x_train, self.y_train)
        return self.XGB

    def train_with_grid(self):
        model = XGBRegressor(random_state=4, tree_method='hist')
        clf = self._grid(model, self.param_grid)
        self.XGB_best = clf.fit(self.x_train, self.y_train)
        self.params_best = self.XGB_best.best_params_
        return self.XGB_best, self.params_best

    def predict(self):
        return self.XGB.predict(self.x_test)

    def predict_with_grid(self):
        return self.XGB_best.predict(self.x_test)
    
class KNNRegressionModel(_BaseRegressor):
    KNN = None
    KNN_best = None
    params_best: Optional[dict] = None
    param_grid = [{
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2],  # 1: Manhattan, 2: Euclidean
        'metric': ['minkowski'],
    }]

    def train(self):
        self.KNN = KNeighborsRegressor(n_neighbors=5, weights='distance').fit(self.x_train, self.y_train)
        return self.KNN

    def train_with_grid(self):
        model = KNeighborsRegressor()
        clf = self._grid(model, self.param_grid)
        self.KNN_best = clf.fit(self.x_train, self.y_train)
        self.params_best = self.KNN_best.best_params_
        return self.KNN_best, self.params_best

    def predict(self):
        return self.KNN.predict(self.x_test)

    def predict_with_grid(self):
        return self.KNN_best.predict(self.x_test)

