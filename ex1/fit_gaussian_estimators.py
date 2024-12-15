from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from utils import make_subplots

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    # *******************************************
    mu, sigma, N = 10, 1, 1000
    uni_gauss = UnivariateGaussian()
    samples = np.random.normal(mu, sigma, N)
    fitted = uni_gauss.fit(samples)
    print(fitted.mu_, fitted.var_)
    # *******************************************

    # Question 2 - Empirically showing sample mean is consistent
    # *******************************************
    X2 = np.linspace(0, 1000, 100, dtype=int)
    Y2 = [np.abs(uni_gauss.fit(samples[:x]).mu_ - 10) for x in X2]

    fig = make_subplots(rows=1, cols=1) \
        .add_traces([go.Scatter(x=X2, y=Y2, mode='lines', marker=dict(color="black"), showlegend=False)]) \
        .update_layout(title_text=r"$\text{(1) Generating Data From Model}$", height=300)

    fig.show()
    # *******************************************


    # Question 3 - Plotting Empirical PDF of fitted model
    # *******************************************
    X3 = samples
    Y3 = fitted.pdf(X3)
    fig = make_subplots(rows=1, cols=1) \
        .add_traces([go.Scatter(x=X3, y=Y3, mode='markers', marker=dict(color="red"))]) \
        .update_layout(title_text=r"$\text{(1) Generating Data From Model}$", height=300)

    fig.show()
    # *******************************************


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    # *******************************************
    N = 1000
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])

    multi_gauss = MultivariateGaussian()
    samples = np.random.multivariate_normal(mu, sigma, N)
    fitted = multi_gauss.fit(samples)
    print(fitted.mu_)
    print(fitted.cov_)
    # *******************************************


    # Question 5 - Likelihood evaluation
    F1 = np.linspace(-10, 10, 200)
    F3 = np.linspace(-10, 10, 200)
    Z = np.zeros((200, 200))
    for i, f1 in enumerate(F1):
        for j, f3 in enumerate(F3):
            Z[i][j] = multi_gauss.log_likelihood(np.array([f1, 0, f3, 0]), sigma, samples)

    heatmap = go.Figure()
    heatmap.add_heatmap(x=F1, y=F3, z=Z)
    heatmap.show()

    # Question 6 - Maximum likelihood
    sum = 0
    max_sum, f1_max, f3_max = -np.inf, 0, 0
    for i, f1 in enumerate(F1):
        for j, f3 in enumerate(F3):
            sum = Z[i, j]
            if sum > max_sum:
                f1_max, f3_max = f1, f3
                max_sum = sum

    print(round(f1_max, 3))
    print(round(f3_max, 3))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
