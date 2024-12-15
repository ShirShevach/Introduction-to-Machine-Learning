from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    errors_train = np.zeros(cv)
    errors_test = np.zeros(cv)

    array_X = np.array_split(X, cv)
    array_y = np.array_split(y, cv)

    for i in range(cv):
        X_test, y_test = array_X[i], array_y[i]
        temp = np.delete(array_X, i, 0)
        X_train = np.concatenate(temp)
        temp = np.delete(array_y, i, 0)
        y_train = np.concatenate(temp)
        # X_train = np.concatenate(([array_X[j] for j in set(range(i)).union(range(i+1, cv))]))
        # y_train = np.concatenate(([array_y[j] for j in set(range(i)).union(range(i+1, cv))]))

        estimator.fit(X_train, y_train)
        y_train_predict = estimator.predict(X_train)
        y_test_predict = estimator.predict(X_test)

        errors_train[i] = scoring(y_train, y_train_predict)
        errors_test[i] = scoring(y_test, y_test_predict)

    mean_error_train = np.mean(errors_train)
    mean_error_test = np.mean(errors_test)

    return mean_error_train, mean_error_test



