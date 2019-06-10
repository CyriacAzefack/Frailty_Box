import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


def main():
    activity = 'meal_preparation'
    raw_dataset = pd.read_csv('./data/{}_dataset.csv'.format(activity)).head(30)

    nb_tstep = len(raw_dataset.columns)

    dataset = raw_dataset.values.flatten()

    ratio = 0.9
    train_size = int(len(dataset) * ratio)

    train, test = dataset[:train_size], dataset[train_size:]

    # autocorrelation_plot(dataset)
    # plt.show()

    # train = np.log(train)
    # train = np.nan_to_num(train)

    # model = ARIMA(train, order=(nb_tstep, 1, nb_tstep))
    model = SARIMAX(train, order=(4, 1, 4), seasonal_order=(1, 1, 1, nb_tstep), enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(len(test))
    # forecast = np.exp(forecast)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(train_size, train_size + len(forecast)), forecast, 'r')
    plt.plot(dataset, 'b')
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
