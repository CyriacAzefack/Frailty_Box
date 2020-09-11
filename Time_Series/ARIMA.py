import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.varmax import VARMAX

sns.set_style('darkgrid')
sns.set(font_scale=1.8)


def main():
    activity = 'meal_preparation'
    raw_dataset = pd.read_csv('./data/{}_duration_distribution.csv'.format(activity), sep=';').drop(['tw_id'],
                                                                                                    axis=1) / 60

    start_date = dt.datetime(2010, 11, 4)
    raw_dataset.index = pd.date_range(start_date, periods=len(raw_dataset), freq='1D')
    #
    # sns.lineplot(data=raw_dataset, palette="tab10", linewidth=2.5, style='event', markers=True, dashes=False)
    # plt.xlabel('Date')
    # plt.ylabel('Durée (en heures)')
    # plt.show()

    # nb_tstep = len(raw_dataset.columns)

    dataset = raw_dataset[['mean', 'std']]
    # dataplus = raw_dataset['std'].values

    ratio = 0.95
    train_size = int(len(dataset) * ratio)

    train, test = dataset[:train_size], dataset[train_size:]
    # trainplus, testplus = dataplus[:train_size], dataset[train_size:]
    # #
    # plt.plot(train)
    # plt.show()

    # train = np.asarray(train).reshape(len(train), 1)
    # trainplus = np.asarray(trainplus).reshape(len(trainplus), 1)

    # diff_train = pd.Series(train['std'])
    #
    # diff_train = diff_train.diff(periods=1)[1:]
    # diff_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    #
    # plt.plot(diff_train)
    # plt.show()
    #
    # diff_train = diff_train.fillna(0)
    #
    # diff_train = diff_train.diff(periods=1)[1:]
    # diff_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    #
    # diff_train = diff_train.fillna(0)
    #
    # # train = diff_train
    #
    # diff_test = pd.Series(np.log(test))
    #
    # diff_test = diff_test.diff(periods=nb_tstep)[nb_tstep:]
    # diff_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    #
    # diff_test = diff_test.fillna(0)
    #
    # test = diff_test
    #
    # plt.plot(diff_train)
    # plt.show()
    # #
    # lag_acf = acf(diff_train, nlags=60)
    # lag_pacf = pacf(diff_train, nlags=60, method='ols')
    #
    # # Plot ACF:
    # plt.figure(figsize=(15, 5))
    # plt.subplot(121)
    # plt.stem(lag_acf)
    # plt.axhline(y=0, linestyle='-', color='black')
    # plt.axhline(y=-1.96 / np.sqrt(len(diff_train)), linestyle='--', color='gray')
    # plt.axhline(y=1.96 / np.sqrt(len(diff_train)), linestyle='--', color='gray')
    # plt.xlabel('Lag')
    # plt.ylabel('ACF')
    #
    # # Plot PACF :
    # plt.subplot(122)
    # plt.stem(lag_pacf)
    # plt.axhline(y=0, linestyle='-', color='black')
    # plt.axhline(y=-1.96 / np.sqrt(len(diff_train)), linestyle='--', color='gray')
    # plt.axhline(y=1.96 / np.sqrt(len(diff_train)), linestyle='--', color='gray')
    # plt.xlabel('Lag')
    # plt.ylabel('PACF')
    #
    # plt.tight_layout()
    # plt.show()

    model = VARMAX(train.values, order=(3, 2), enforce_stationarity=True, enforce_invertibility=False)
    #
    # model = SARIMAX(train, order=(2, 1, 3), seasonal_order=(0, 0, 0, 0), enforce_stationarity=True,
    #                 enforce_invertibility=False)

    # model = ARIMA(train, order=(2, 0, 2))
    model_fit = model.fit(disp=False)

    print(model_fit.summary())
    forecast = model_fit.forecast(len(test))

    forecast_df = pd.DataFrame(forecast, test.index, ['mean_forecast', 'std_forecast'])

    # Add last row from train to forecast, to link in the graph
    new_row = pd.DataFrame({
        'mean_forecast': train['mean'].values[-1],
        'std_forecast': train['std'].values[-1]
    },
        index=[train.index[-1]]
    )
    # simply concatenate both dataframes
    forecast_df = pd.concat([new_row, forecast_df])

    # forecast = np.exp(forecast)

    # plt.figure(figsize=(10, 5))
    #
    sns.lineplot(data=raw_dataset, palette="tab10", linewidth=2.5, style='event', markers=True, dashes=False)
    sns.lineplot(data=forecast_df, palette="prism", linewidth=2.5, style='event', markers=True, dashes=False)

    #
    #
    # sns.lineplot(x=np.arange(train_size, train_size + len(forecast)), y=forecast, label='Forecast')
    # sns.lineplot(x=np.arange(len(dataset)), y=dataset, alpha=0.6,  label='Training Data')
    # # plt.plot(dataset, 'b', alpha=0.6)
    error = mean_squared_error(test, forecast) / np.mean(test)
    # plt.title('Test NMSE: %.3f' % error)
    plt.xlabel('Date')
    plt.ylabel('Durée')
    plt.axvline(x=train.index[-1], color='black')
    plt.show()

    # print('Test NMSE: %.3f' % error)
    #
    # # plot
    # plt.plot(test)
    # plt.plot(predictions, color='red')
    # plt.show()


if __name__ == "__main__":
    main()
