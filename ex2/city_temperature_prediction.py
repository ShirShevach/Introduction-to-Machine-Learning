import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import plotly.graph_objects as go


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # raise NotImplementedError()
    data = pd.read_csv(filename, parse_dates=["Date"]).dropna()
    data["DayOfYear"] = data["Date"].dt.dayofyear
    for i, value in data["Temp"].iteritems():
        if value <= -50 or value > 50:
            data.drop([i], inplace=True)
    return data


def split_train_test_poly(X, y, train_proportion):
    relative_num = int(X.shape[0] * train_proportion)
    train_X = X.iloc[:relative_num]
    train_y = y[:relative_num]
    test_X = X.iloc[relative_num:]
    test_y = y.iloc[relative_num:]
    return train_X, train_y, test_X, test_y


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("C:/Users/ASUS/Desktop/IML.HUJI/datasets/City_Temperature.csv")
    israel_data = data.groupby(["Country"]).get_group("Israel")

    # Question 2 - Exploring data for specific country
    # part 1:
    x = israel_data.groupby("Year")
    YearOfDay_lst = []
    Temp_lst = []
    for i in range(13):
        curr_year = x.get_group(1995+i)
        YearOfDay_lst.append(curr_year["DayOfYear"])
        Temp_lst.append(curr_year["Temp"])

    fig = go.Figure([go.Scatter(x=YearOfDay_lst[0], y=Temp_lst[0], mode="markers", name="1995",
                                line=dict(color="black", width=1)),
                     go.Scatter(x=YearOfDay_lst[1], y=Temp_lst[1], mode="markers", name="1996",
                                line=dict(color="blue", width=1)),
                    go.Scatter(x=YearOfDay_lst[2], y=Temp_lst[2], mode="markers", name="1997",
                               line=dict(color="red", width=1)),
                    go.Scatter(x=YearOfDay_lst[3], y=Temp_lst[3], mode="markers", name="1998",
                               line=dict(color="yellow", width=1)),
                    go.Scatter(x=YearOfDay_lst[4], y=Temp_lst[4], mode="markers", name="1999",
                               line=dict(color="orange", width=1)),
                    go.Scatter(x=YearOfDay_lst[5], y=Temp_lst[5], mode="markers", name="2000",
                               line=dict(color="lightcoral", width=1)),
                    go.Scatter(x=YearOfDay_lst[6], y=Temp_lst[6], mode="markers", name="2001",
                               line=dict(color="darkorchid", width=1)),
                    go.Scatter(x=YearOfDay_lst[7], y=Temp_lst[7], mode="markers", name="2002",
                               line=dict(color="cadetblue", width=1)),
                    go.Scatter(x=YearOfDay_lst[8], y=Temp_lst[8], mode="markers", name="2003",
                               line=dict(color="cyan", width=1)),
                    go.Scatter(x=YearOfDay_lst[9], y=Temp_lst[9], mode="markers", name="2004",
                               line=dict(color="skyblue", width=1)),
                    go.Scatter(x=YearOfDay_lst[10], y=Temp_lst[10], mode="markers", name="2005",
                               line=dict(color="gold", width=1)),
                    go.Scatter(x=YearOfDay_lst[11], y=Temp_lst[11], mode="markers", name="2006",
                               line=dict(color="hotpink", width=1)),
                    go.Scatter(x=YearOfDay_lst[12], y=Temp_lst[12], mode="markers", name="2007",
                               line=dict(color="purple", width=1))],
                    layout=go.Layout(title=f"Graph temperature in day-of-year on Israel",
                                     xaxis={"title": "day of year"},
                                     yaxis={"title": "temperatures"},
                                     height=400))
    fig.show()

    # part 2:
    israel_data = data.groupby(["Country"]).get_group("Israel")
    A = israel_data.groupby("Month")["Temp"].agg(np.std)
    fig2 = px.bar(A)
    fig2.show()

    # Question 3 - Exploring differences between countries
    q3_data = data.groupby(['Country', 'Month']).agg(mean_temp=('Temp', np.mean), std_temp=('Temp', np.std))
    q3_data.reset_index(inplace=True)
    fig3 = px.line(q3_data, x='Month', y='mean_temp', error_y='std_temp', color='Country')
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    loss_over_k = []
    train_X, train_y, test_X, test_y = split_train_test_poly(israel_data["DayOfYear"], israel_data["Temp"], 0.25)
    for k in range(1, 11):
        polynomial_fit = PolynomialFitting(k)
        polynomial_fit._fit(train_X.to_numpy(), train_y.to_numpy())
        loss_over_k.append(polynomial_fit._loss(test_X.to_numpy(), test_y.to_numpy()))
        print(np.round(loss_over_k[-1]), 2)

    fig4 = px.bar(loss_over_k)
    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    min_k = np.argmin(loss_over_k) + 1
    polynomial_fit_q5 = PolynomialFitting(min_k)
    polynomial_fit_q5._fit(train_X.to_numpy(), train_y.to_numpy())
    loss_over_countries = []
    countries = ["South Africa", "The Netherlands", "Israel", "Jordan"]
    for country in countries:
        country_data = data.groupby(["Country"]).get_group(country)
        loss_over_countries.append(polynomial_fit_q5._loss(country_data["DayOfYear"], country_data["Temp"]))

    fig5 = px.bar(x=countries, y=loss_over_countries,
                  title="The average loss of temperature forecasting of countries based on study israel")
    fig5.show()