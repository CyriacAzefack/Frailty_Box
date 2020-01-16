import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima.arima.stationarity import ADFTest
from pmdarima.arima.utils import ndiffs, nsdiffs
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace import sarimax


def main():
    activity = 'meal_preparation'
    # raw_dataset = pd.read_csv('./data/{}_dataset.csv'.format(activity))

    raw_dataset = pd.read_csv(f'data/{activity}_duration_distribution.csv', sep=";")['mean']

    season = 7
    ratio = 0.9

    dataset = pd.Series(raw_dataset.values.flatten())

    arima_forecast(data=dataset, train_ratio=ratio, seasonality=season, nb_steps_to_forecast=50, display=True)


def arima_forecast(data, train_ratio, seasonality, nb_steps_to_forecast, display=False):
    """
    Train an ARIMA model to forecast the data
    :param data:
    :param train_ratio:
    :param seasonality:
    :return:
    """
    train_size = int(len(data) * train_ratio)

    data = pd.Series(data)

    train, test = data[:train_size], data[train_size:]

    plt.plot(data)
    plt.show()
    test_size = test.shape[0]

    # Find the parameters
    # Estimate the number of differences using an ADF test:
    d = ndiffs(train, test='adf')  # -> 0

    if display: print(f'Estimated d = {d}')

    # estimate number of seasonal differences using a Canova-Hansen test
    D = nsdiffs(train,
                m=seasonality,  # commonly requires knowledge of dataset
                max_D=10,
                test='ch')  # -> 0
    if display: print(f"Estimated 'D' = {D}")

    sarima_model = pm.auto_arima(train,
                                 start_p=0, max_p=3,
                                 start_q=0, max_q=3,
                                 start_P=0, max_P=3,
                                 start_Q=0, max_Q=3,
                                 m=seasonality, seasonal=True,
                                 d=d,
                                 max_iter=50,
                                 method='lbfgs',
                                 trace=True,
                                 n_jobs=-1,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True)  # set to stepwise

    # raw_forecast = sarima_model.predict(test_size + nb_steps_to_forecast)

    p, d, q = sarima_model.order
    P, D, Q, m = sarima_model.seasonal_order

    sarima_model = sarimax.SARIMAX(train, trends='ct', order=(p, d, q), seasonal_order=(P, D, Q, seasonality))
    sarima_model = sarima_model.fit(disp=False)

    all_forecast = sarima_model.predict(start=0, end=train_size + test_size + nb_steps_to_forecast)

    all_forecast = pd.Series(all_forecast)
    # raw_forecast = sarima_model.get_forecast(
    #     steps=test_size + nb_steps_to_forecast).predicted_mean  # predict N steps into the future

    raw_forecast = all_forecast[train_size:]

    validation_forecast = raw_forecast[:test_size]
    nmse_error = mean_squared_error(test, validation_forecast) / np.mean(test)

    # Forecast to use
    forecasts = raw_forecast[test_size:]

    if display:
        # Visualize the forecasts (blue=train, green=forecasts)
        print(sarima_model.summary())
        plt.plot(train, label='Training data')
        plt.plot(test, label='Test data')
        # plt.plot(np.arange(train_size, train_size + len(raw_forecast)), raw_forecast, label='Predicted data')
        plt.plot(all_forecast, label='Predicted data')
        plt.title(f'NMSE ERROR : {nmse_error:.3f}')
        plt.legend()
        plt.show()

    return mse_error, forecasts


if __name__ == '__main__':
    main()
