# VARMA example
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR

from Utils import *

sns.set_style('darkgrid')


def main():
    activity = 'sleeping'
    dataset = pd.read_csv('{}_dataset.csv'.format(activity))

    desc = dataset.describe().T
    const_cols = desc[desc['std'] == 0].index

    # remove constant columns from analysis
    dataset.drop(const_cols, axis=1, inplace=True)

    lag = 1
    dataset_transformed = dataset.copy()

    for col in dataset_transformed.columns:
        dataset_transformed[col] = np.log(dataset.loc[:, col]).diff(lag)

    dataset_transformed.dropna(inplace=True)
    per_train = 0.6

    nb_train = int(per_train * (len(dataset)))
    train_df = dataset_transformed.head(nb_train)

    # apply adf test on the series
    # adf_test(dataset['Ts_0'])

    # Check constant columns

    nb_feat = len(dataset_transformed.columns)

    data = np.asarray(dataset_transformed)

    # data = data[:, 0:24]

    # check stationarity
    # print('## STATIONARITY ##')
    # print(coint_johansen(data, -1, 1).eig)

    train = data[:nb_train]
    valid = data[nb_train:]

    # fit model
    model = VAR(endog=data)
    model_fit = model.fit()

    # make prediction
    prediction = model_fit.forecast(model_fit.y, steps=len(valid))

    # converting predictions to dataframe
    cols = ['TS_{}'.format(i) for i in range(nb_feat)]
    pred = pd.DataFrame(index=range(0, len(prediction)), columns=[cols])

    for j in range(nb_feat):
        for i in range(0, len(prediction)):
            pred.iloc[i][j] = prediction[i][j]

    # check rmse
    rmse_values = []
    for col in cols:
        rmse = math.sqrt(mean_squared_error(pred[[col]].values, valid[:, cols.index(col)]))
        rmse_values.append(rmse)
        # print('rmse value for {} is : {}'.format(col, rmse))

    plt.hist(rmse_values)
    plt.title('RMSE Histograms for Time Series')
    plt.show()

    base_real = np.arange(len(train))
    base_forecast = np.arange(len(train), len(train) + len(valid))

    fig, axes = plt.subplots(10, 1)

    for i in range(10):
        axes[i].plot(base_real, train[:, i], c='blue', label='train')
        axes[i].plot(base_forecast, pred[cols[i]].values, c='red', label='forecast')
        axes[i].plot(base_forecast, valid[:, i], c='green', label='valid')
        axes[i].set_title(cols[i])

    plt.legend()
    plt.show()
    # print(yhat)


def adf_test(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


if __name__ == '__main__':
    main()
