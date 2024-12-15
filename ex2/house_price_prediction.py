from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # raise NotImplementedError()
    data = pd.read_csv(filename).dropna()
    for i, value in data["price"].iteritems():
        if value <= 0:
            data.drop([i], inplace=True)
    y = data["price"]

    data["date"] = pd.to_datetime(data["date"], infer_datetime_format=True).apply(lambda x: x.value)

    # edit the year_renovated col
    data["yr_renovated"] = np.where(data["yr_renovated"] != 0, data["yr_renovated"], data["yr_built"])

    # add a col "is_basement"
    data["is_basement"] = np.where(data["sqft_basement"] > 0, 1, 0)

    # make the basement be categorical
    data["below_500_basement"] = np.where(
        np.logical_and(data["sqft_basement"] <= 500, data["sqft_basement"] > 0), 1, 0)
    data["above_500_till_1000_basement"] = np.where(
        np.logical_and(data["sqft_basement"] > 500, data["sqft_basement"] < 1000), 1, 0)
    data["above_1000_basement"] = np.where(data["sqft_basement"] >= 1000, 1, 0)

    data.pop("sqft_basement")
    data.pop("id")
    data.pop("price")
    data.pop("zipcode")
    return data, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for key, value in X.iteritems():
        p = (np.cov(value.array, y) / (np.std(value.array) * np.std(y)))[0][1]
        fig = make_corr_graph(value.array, y, p, key)
        plotly.offline.plot(fig, filename=output_path + f"\\{key}.html", auto_open=False)


def make_corr_graph(x, y, corr, name):
    fig = go.Figure([go.Scatter(x=x, y=y, mode="markers",
                                line=dict(color="black", width=1))],
                    layout=go.Layout(title=f"Graph of the feature '{name}' with the correlation {corr}",
                                     xaxis={"title": f"x - {name}"},
                                     yaxis={"title": "y - Response"},
                                     height=400))
    return fig


def make_graph_mean_loss(x, y_mean, y_var_top, y_var_below):
    fig = go.Figure([
        go.Scatter(
            name='mean loss',
            x=x,
            y=y_mean,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='var top',
            x=x,
            y=y_var_top,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='var low',
            x=x,
            y=y_var_below,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        yaxis_title='p',
        title='mean loss and std of the loss',
        hovermode="x"
    )
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("C:/Users/ASUS/Desktop/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "C:/Users/ASUS/Desktop")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.25)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    mean_loss_list = []
    y_var_top = []
    y_var_below = []
    for p in range(10, 101):
        ten_loss = []
        linear_regress = LinearRegression()
        for i in range(10):
            samples_X = train_X.sample(frac=p / 100)
            samples_y = train_y.iloc[samples_X.index]
            linear_regress.fit(samples_X.to_numpy(), samples_y)
            loss = linear_regress.loss(test_X.to_numpy(), test_y.to_numpy())
            ten_loss.append(loss)

        curr_mean_loss = np.mean(ten_loss)
        curr_var_loss = np.std(ten_loss)
        mean_loss_list.append(curr_mean_loss)
        y_var_top.append(curr_mean_loss + 2*curr_var_loss)
        y_var_below.append(curr_mean_loss - 2*curr_var_loss)

    row_p = np.array(range(10, 101))
    make_graph_mean_loss(row_p, mean_loss_list, y_var_top, y_var_below)

