import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

sns.set_style('darkgrid')
np.random.seed(7)


def main():
    activity = 'sleeping'
    raw_dataset = pd.read_csv('{}_dataset.csv'.format(activity))
    #
    # raw_dataset = raw_dataset[['Ts_0']]

    look_back = 24

    values = []
    for id, row in raw_dataset.iterrows():
        values += list(row.values)

    dataset = pd.DataFrame(values, columns=['count'])
    # dataset['ts'] = dataset.index

    dataset = dataset.values

    plt.plot(dataset)
    plt.show()

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print(len(train), len(test))

    # reshape into X=t and Y=t+1
    trainX, trainY = create_lookback_dataset(train, look_back)
    testX, testY = create_lookback_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(20, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='RMSprop')
    history = model.fit(trainX, trainY, epochs=50, batch_size=20, validation_data=(testX, testY), verbose=2)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0])) / np.mean(trainY[0])
    print('Train Percentage of relative error: {:.2f}'.format(trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0])) / np.mean(testY[0])
    print('Test Percentage of relative error: {:.2f}'.format(testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)

    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('Entropy loss')
    plt.legend()
    plt.show()


# convert an array of values into a dataset matrix
def create_lookback_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.asarray(dataX), np.asarray(dataY)


if __name__ == "__main__":
    main()
