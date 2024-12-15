from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """

    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for name, file in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{file}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(percep, X_i, y_i):
            losses.append(percep.loss(X, y))

        perceptron = Perceptron(callback=callback)
        perceptron.fit(X, y)
        iters = np.array(range(1, len(losses)+1))
        # Plot figure of loss as function of fitting iteration

        fig = go.Figure([go.Scatter(x=iters, y=losses, mode="lines",
                                    line=dict(color="black", width=1))],
                        layout=go.Layout(title=f"Graph {name}",
                                         xaxis={"title": f"x - num of iterations"},
                                         yaxis={"title": "y - losses"},
                                         height=400))
        fig.show()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for file in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{file}")

        # Fit models and predict over training set
        lda = LDA()
        gnb = GaussianNaiveBayes()
        lda.fit(X, y)
        gnb.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        y_pred_gnb = gnb.predict(X)
        y_pred_lda = lda.predict(X)

        x1 = np.array(X[:, 0])
        x2 = np.array(X[:, 1])
        acc_gnb = accuracy(y_pred_gnb, y)
        acc_lda = accuracy(y_pred_lda, y)

        # Add traces for data-points setting symbols and colors
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"GraphGaussianNaive, accuracy {acc_gnb}",
                                                            f"LDA, accuracy {acc_lda}"))
        fig.append_trace(go.Scatter(x=x1, y=x2, mode="markers",
                                    marker=dict(color=y_pred_gnb, symbol=y)),
                         row=1, col=1)

        fig.append_trace(go.Scatter(x=x1, y=x2, mode="markers",
                                    marker=dict(color=y_pred_lda, symbol=y)),
                         row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.append_trace(go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1], mode="markers",
                                    marker=dict(color="black", symbol='x', size=12)), row=1, col=1)
        fig.append_trace(go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers",
                                    marker=dict(color="black", symbol='x', size=12)), row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i, k in enumerate(gnb.classes_):
            fig.append_trace(get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])), row=1, col=1)
            fig.append_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=2)

        fig.update_layout(title=file[:-4])
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    # # q1
    # # S = {(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 2), (7, 2)}
    # x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    # y = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    # naive_bayes = GaussianNaiveBayes()
    # naive_bayes.fit(x, y)
    # print("pi q1")
    # print(naive_bayes.pi_)
    # print("mu q1")
    # print(naive_bayes.mu_)
    # print("\n\n")
    #
    # # q2
    # # S = {([1, 1], 0), ([1, 2], 0), ([2, 3], 1), ([2, 4], 1), ([3, 3], 1), ([3, 4], 1)}
    # x = np.array(
    #     [np.array([1, 1]), np.array([1, 2]), np.array([2, 3]), np.array([2, 4]), np.array([3, 3]), np.array([3, 4])])
    # y = np.array([0, 0, 1, 1, 1, 1])
    # naive_bayes2 = GaussianNaiveBayes()
    # naive_bayes2.fit(x, y)
    # print("pi q2")
    # print(naive_bayes2.pi_)
    # print("mu q2")
    # print(naive_bayes2.mu_)
    # print("vars q2")
    # print(naive_bayes2.vars_)

    compare_gaussian_classifiers()
