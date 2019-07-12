import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
# ARMA example

from statsmodels.tsa.stattools import acf, pacf


def main():
    activity = 'sleeping'
    raw_dataset = pd.read_csv('./data/{}_dataset.csv'.format(activity))

    nb_tstep = len(raw_dataset.columns)

    dataset = raw_dataset.values.flatten()

    ratio = 0.95
    train_size = int(len(dataset) * ratio)

    train, test = dataset[:train_size], dataset[train_size:]
    # #
    # plt.plot(train)
    # plt.show()

    diff_train = pd.Series(np.log(train))

    diff_train = diff_train.diff(periods=nb_tstep)[nb_tstep:]
    diff_train.replace([np.inf, -np.inf], np.nan, inplace=True)

    diff_train = diff_train.fillna(0)

    # # train = diff_train
    #
    # diff_test = pd.Series(np.log(test))
    #
    # diff_test = diff_test.diff(periods=nb_tstep)[nb_tstep:]
    # diff_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    #
    # diff_test = diff_test.fillna(0)

    # test = diff_test
    #
    # plt.plot(diff_train)
    # plt.show()
    #
    lag_acf = acf(diff_train, nlags=4 * nb_tstep)
    lag_pacf = pacf(diff_train, nlags=4 * nb_tstep, method='ols')

    # Plot ACF:
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.stem(lag_acf)
    plt.axhline(y=0, linestyle='-', color='black')
    plt.axhline(y=-1.96 / np.sqrt(len(diff_train)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(diff_train)), linestyle='--', color='gray')
    plt.xlabel('Lag')
    plt.ylabel('ACF')

    # Plot PACF :
    plt.subplot(122)
    plt.stem(lag_pacf)
    plt.axhline(y=0, linestyle='-', color='black')
    plt.axhline(y=-1.96 / np.sqrt(len(diff_train)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(diff_train)), linestyle='--', color='gray')
    plt.xlabel('Lag')
    plt.ylabel('PACF')

    plt.tight_layout()
    plt.show()
    #



    # autocorrelation_plot(dataset)
    # plt.show()

    # train = np.log(train)
    # train = np.nan_to_num(train)

    # model = ARIMA(train, order=(nb_tstep, 1, nb_tstep))
    model = SARIMAX(train, order=(3, 0, 0), seasonal_order=(2, 1, 0, nb_tstep), enforce_stationarity=False,
                    enforce_invertibility=False)

    # model = ARIMA(train, order=(2, 0, 2))
    model_fit = model.fit(disp=False)

    print(model_fit.summary())
    forecast = model_fit.forecast(len(test))
    # forecast = np.exp(forecast)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(train_size, train_size + len(forecast)), forecast, 'r')
    plt.plot(dataset, 'b', alpha=0.6)
    error = mean_squared_error(test, forecast) / np.mean(test)
    plt.title('Test NMSE: %.3f' % error)
    plt.xlabel('Time')
    plt.ylabel('count')
    plt.axvline(x=train_size, color='black')
    plt.show()

    print('Test NMSE: %.3f' % error)
    #
    # # plot
    # plt.plot(test)
    # plt.plot(predictions, color='red')
    # plt.show()


if __name__ == "__main__":
    main()
