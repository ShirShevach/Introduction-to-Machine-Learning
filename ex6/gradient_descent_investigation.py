from copy import deepcopy

import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule, BaseEstimator
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2, RegularizedModule, LogisticModule
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve, auc

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values_list, weights_list = [], []

    def call_back(**kwargs):
        values_list.append(kwargs["val"])
        weights_list.append(kwargs["weights"])
    return call_back, values_list, weights_list


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    x = np.array(range(1000))
    for norm in (L1, L2):
        q3 = go.Figure()
        for eta in etas:
            callback, values_list, weights_list = get_gd_state_recorder_callback()
            gradient_descent = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            gradient_descent.fit(norm(init), X=None, y=None)
            descent_path = np.r_[weights_list]
            q1 = plot_descent_path(module=norm, descent_path=descent_path, title=f"for {norm.__name__}, eta={eta}")
            q1.write_image(f"Q1 {norm.__name__}, eta={eta}.png")
            q3.add_trace(go.Scatter(x=x, y=values_list, mode="lines", name=f"{norm.__name__}, eta={eta}"))
        q3.update_layout(title=f"Graph Q3 {norm.__name__}")
        q3.write_image(f"Q3 {norm.__name__}.png")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    x = np.array(range(1000))
    q5 = go.Figure()
    for gamma in gammas:
        callback, values_list, weights_list = get_gd_state_recorder_callback()
        gradient_descent = GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback)
        gradient_descent.fit(L1(init), X=None, y=None)
        descent_path = np.r_[weights_list]
        if gamma == .95:
            q6 = plot_descent_path(module=L1, descent_path=descent_path,
                                   title=f"for {L1.__name__}, eta={eta}, gamma={gamma}")
            q6.write_image(f"Q6 {L1.__name__}, eta={eta}, gamma={gamma}.png")
        print(f"for {L1.__name__}, gamma={gamma}")
        q5.add_trace(go.Scatter(x=x, y=values_list, mode="lines", name=f"{L1.__name__}, eta={eta}, gamma={gamma}"))

    # Plot algorithm's convergence for the different values of gamma
    q5.update_layout(title=f"Graph Q5 {L1.__name__}")
    q5.write_image(f"Q5 {L1.__name__}.png")

    # Plot descent path for gamma=0.95


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring, cv: int = 5) -> Tuple[float, float]:
    ids = np.arange(X.shape[0])

    # Randomly split samples into `cv` folds
    folds = np.array_split(ids, cv)

    train_score, validation_score = .0, .0
    for fold_ids in folds:
        train_msk = ~np.isin(ids, fold_ids)
        fit = deepcopy(estimator).fit(X[train_msk], y[train_msk])

        train_score += scoring(y[train_msk], fit.predict(X[train_msk]))
        validation_score += scoring(y[fold_ids], fit.predict(X[fold_ids]))

    return train_score / cv, validation_score / cv


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train.to_numpy(), y_train.to_numpy())
    y_predict_proba = logistic_regression.predict_proba(X_train.to_numpy())

    fpr, tpr, thresholds = roc_curve(y_train, y_predict_proba)
    q8 = go.Figure(
    data=[go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(color="black", dash='dash'), name="Random Class Assignment"),
          go.Scatter(x=fpr, y=tpr, mode='markers+lines',text=thresholds, name="", showlegend=False, marker_size=5,
                     hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
    layout=go.Layout(title=rf"$\text{{Q8 ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                                 xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                                 yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    q8.write_image(f"q8.png")
    values_alphas = tpr - fpr
    max_val = np.argmax(values_alphas)
    print("best alpha: ", thresholds[max_val])

    lambadas = [.001, .002, .005, .01, .02, .05, .1]
    min_error_test = np.inf
    min_lambda = -1
    # L1:
    for lam in lambadas:
        l1_reg_logist = LogisticRegression(penalty='l1', lam=lam)
        train_score, validation_score = cross_validate(l1_reg_logist, X_train.to_numpy(), y_train.to_numpy()
                                                       , mean_square_error)
        if validation_score < min_error_test:
            min_error_test = validation_score
            min_lambda = lam

    print(f"For L1, the minimum lambada is {min_lambda}, its test error is {min_error_test}")

    # L2:
    min_error_test = np.inf
    min_lambda = -1
    for lam in lambadas:
        l2_reg_logist = LogisticRegression(penalty='l2', lam=lam)
        train_score, validation_score = cross_validate(l2_reg_logist, X_train.to_numpy(), y_train.to_numpy()
                                                       , mean_square_error)
        if validation_score < min_error_test:
            min_error_test = validation_score
            min_lambda = lam

    print(f"For L2, the minimum lambada is {min_lambda}, its test error is {min_error_test}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
