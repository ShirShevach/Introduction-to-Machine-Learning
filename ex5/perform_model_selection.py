from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def f(x):
    return (x+3)*(x+2)*(x+1)*(x-1)*(x-2)

def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X_true = np.linspace(-1.2, 2, n_samples)
    y_true = [f(X_true[i]) for i in range(n_samples)]
    X = np.random.uniform(low=-1.2, high=2, size=n_samples)
    y = [f(X[i]) + np.random.normal(0, noise) for i in range(n_samples)]

    X, y = pd.DataFrame(X), pd.DataFrame(y)
    train_X, train_y, test_X, test_y = split_train_test(X, y, 2/3)
    X = X[0].to_numpy()
    train_X, train_y = train_X[0].to_numpy(), train_y[0].to_numpy()
    test_X, test_y = test_X[0].to_numpy(), test_y[0].to_numpy()
    fig1 = go.Figure([go.Scatter(x=X_true, y=y_true, mode="lines", name="true",
                                 line=dict(color="black", width=1)),
                      go.Scatter(x=train_X, y=train_y, mode="markers", name="train",
                                 line=dict(color="blue", width=1)),
                      go.Scatter(x=test_X, y=test_y, mode="markers", name="test",
                                line=dict(color="red", width=1))],
                     layout=go.Layout(title=f"Graph Q1 - noise is {noise}",
                                      xaxis={"title": ""},
                                      yaxis={"title": ""},
                                      height=400))
    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    array_k = np.array(range(0, 11))
    train_errors = np.zeros(11)
    validation_errors = np.zeros(11)
    for i in range(11):
        train_errors[i], validation_errors[i] = cross_validate(PolynomialFitting(i), train_X, train_y,
                                                               mean_square_error)

    fig2 = go.Figure([go.Scatter(x=array_k, y=train_errors, mode="lines", name="train_errors",
                                 line=dict(color="blue", width=1)),
                      go.Scatter(x=array_k, y=validation_errors, mode="lines", name="validation_errors",
                                 line=dict(color="red", width=1))],
                     layout=go.Layout(title=f"Graph Q2 - Perform CV for polynomial fitting for noise {noise}",
                                      xaxis={"title": ""},
                                      yaxis={"title": ""},
                                      height=400))
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_k = np.argmin(validation_errors)
    poly_fit = PolynomialFitting(min_k)
    poly_fit.fit(train_X, train_y)
    print("min_k", min_k)
    print("test error", poly_fit.loss(test_X, test_y))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    errors_train_lasso = np.zeros(n_evaluations)
    errors_validation_lasso = np.zeros(n_evaluations)
    errors_train_ridge = np.zeros(n_evaluations)
    errors_validation_ridge = np.zeros(n_evaluations)

    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam_array = np.linspace(0.001, 1, n_evaluations)
    for i, k in enumerate(lam_array):
        errors_train_lasso[i], errors_validation_lasso[i] = cross_validate(Lasso(k), train_X, train_y,
                                                                           mean_square_error)
        errors_train_ridge[i], errors_validation_ridge[i] = cross_validate(RidgeRegression(k), train_X, train_y,
                                                                           mean_square_error)

    fig7 = make_subplots(rows=1, cols=2, subplot_titles=["Lasso", "Ridge"])
    fig7.append_trace(go.Scatter(x=lam_array, y=errors_train_lasso, mode="lines", name="errors_train_lasso",
                                 line=dict(color="black", width=1)), row=1, col=1)
    fig7.append_trace(go.Scatter(x=lam_array, y=errors_validation_lasso, mode="lines", name="errors_validation_lasso",
                                 line=dict(color="blue", width=1)), row=1, col=1)

    fig7.append_trace(go.Scatter(x=lam_array, y=errors_train_ridge, mode="lines", name="errors_train_ridge",
                                 line=dict(color="red", width=1)), row=1, col=2)
    fig7.append_trace(go.Scatter(x=lam_array, y=errors_validation_ridge, mode="lines", name="errors_validation_ridge",
                                 line=dict(color="pink", width=1)), row=1, col=2)
    fig7.update_layout(title="Graph Q7")
    fig7.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    min_arg_lasso = np.argmin(errors_validation_lasso)
    min_arg_ridge = np.argmin(errors_validation_ridge)
    min_lam_lasso = lam_array[min_arg_lasso]
    min_lam_ridge = lam_array[min_arg_ridge]
    print("min_lam_lasso", min_lam_lasso)
    print("min_lam_ridge", min_lam_ridge)

    linear_regress = LinearRegression()
    lasso = Lasso(min_lam_lasso)
    ridge = RidgeRegression(min_lam_ridge)
    linear_regress.fit(train_X, train_y)
    lasso.fit(train_X, train_y)
    ridge.fit(train_X, train_y)
    print("error LinearRegression", linear_regress.loss(test_X, test_y))
    y_predict_lasso = lasso.predict(test_X)
    print("error lasso", mean_square_error(test_y, y_predict_lasso))
    print("error ridge", ridge.loss(test_X, test_y))

if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(noise=5)
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)

    select_regularization_parameter()
