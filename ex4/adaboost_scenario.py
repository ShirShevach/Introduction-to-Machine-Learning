import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y

def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_boost = AdaBoost(DecisionStump, n_learners)
    ada_boost._fit(train_X, train_y)

    x = np.array(range(1, n_learners + 1))
    y_training = np.zeros(n_learners)
    y_testError = np.zeros(n_learners)
    for i in range(n_learners):
        y_training[i] = ada_boost.partial_loss(train_X, train_y, i+1)
        y_testError[i] = ada_boost.partial_loss(test_X, test_y, i+1)

    fig1 = go.Figure([go.Scatter(x=x, y=y_training, mode="lines", name="y_training",
                                line=dict(color="blue", width=1)),
                     go.Scatter(x=x, y=y_testError, mode="lines", name="y_testError",
                                line=dict(color="red", width=1))],
                    layout=go.Layout(title=f"Graph Q1 training and test errors, noise={noise}",
                                     xaxis={"title": ""},
                                     yaxis={"title": ""},
                                     height=400))
    fig1.show()


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    if noise == 0:
        fig2 = make_subplots(rows=2, cols=2, subplot_titles=["5", "50", "100", "250"])
        for i, t in enumerate(T):
            fig2.add_traces([decision_surface(lambda X: ada_boost.partial_predict(X, t),
                                              lims[0], lims[1], showscale=False),
                             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                        marker=dict(color=test_y.astype(int),
                                                    symbol=test_y.astype(int) + 2,
                                                    colorscale=[custom[0], custom[-1]],
                                                    line=dict(color="black", width=1)))],
                            rows=(i // 2) + 1, cols=(i % 2) + 1)
        fig2.update_layout(title="Q2: test with 4 iterations")
        fig2.show()

    # Question 3: Decision surface of best performing ensemble
        arg_min = np.argmin(y_testError) + 1
        acurr_of_min = accuracy(test_y, ada_boost.partial_predict(test_X, arg_min))

        fig3 = go.Figure([decision_surface(lambda X: ada_boost.partial_predict(X, i),
                                           lims[0], lims[1], showscale=False),
                          go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                     marker=dict(color=test_y.astype(int),
                                                 symbol=test_y.astype(int) + 2,
                                                 colorscale=[custom[0], custom[-1]],
                                                 line=dict(color="black", width=1)))])
        fig3.update_layout(
            title=f"Q3: Decision Surface of Ensemble with lowest Error is the size {arg_min} with the accuracy is {acurr_of_min}")
        fig3.show()


    # Question 4: Decision surface with weighted samples
    D = ada_boost.D_ / np.max(ada_boost.D_) * 10
    fig4 = go.Figure([decision_surface(lambda X: ada_boost.partial_predict(X, n_learners),
                                       lims[0], lims[1], showscale=False),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=train_y.astype(int),
                                             symbol=train_y.astype(int) + 2,
                                             size=D,
                                             colorscale=[custom[0], custom[-1]],
                                             line=dict(color="black", width=1)))])
    fig4.update_layout(title=f"Q4: Final Decision Boundary with proportional size training data point, noise={noise}")
    fig4.show()

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
