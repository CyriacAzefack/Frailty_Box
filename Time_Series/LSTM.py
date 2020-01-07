import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

sns.set_style('darkgrid')
np.random.seed(7)


def main():
    dataset_name = "aruba"

    # episode = ('bed_to_toilet', 'sleeping')
    episode = ('bed_to_toilet', 'sleeping')

    output = "C:/Users/cyriac.azefack/Workspace/Frailty_Box/output/{}/Activity_{}/".format(dataset_name, episode)

    daily_profile = pickle.load(open(output + "daily_profile.pkl", "rb"))

    duration_distrib = pickle.load(open(output + "durations_distrib.pkl", "rb"))

    occurrence_order = pickle.load(open(output + "execution_order.pkl", "rb"))

    inter_events = pickle.load(open(output + "inter_events.pkl", "rb"))

    names = []
    # data = duration_distrib[episode[0]]['std'].values
    # data = inter_events["lambda"].values
    dataset = duration_distrib[episode[0]].drop(columns=['tw_id']).values

    dataset = inter_events
    names.append('lambda_inter_events')

    for label in episode:
        data = duration_distrib[label]
        dataset = dataset.join(data.set_index('tw_id'), on='tw_id', rsuffix='_' + label)
        names.append('mean_{}'.format(label))
        names.append('std_{}'.format(label))

    dataset = dataset.join(occurrence_order.set_index('tw_id'), on='tw_id')
    names += list(occurrence_order.columns[1:])

    dataset.drop(columns=['tw_id'], inplace=True)

    dataset.columns = names

    n_features = len(names)
    look_back = 1
    train_ratio = 0.8
    test_ratio = 0.1

    # plt.plot(log_dataset)
    # plt.show()

    # normalize the log_dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets

    train, test, validate = np.split(dataset,
                                     [int(train_ratio * len(dataset)), int((train_ratio + test_ratio) * len(dataset))])
    # train_size = int(len(log_dataset) * train_ratio)
    # test_size = len(log_dataset) - train_size
    # train, test = log_dataset[0:train_size, :], log_dataset[train_size:len(log_dataset), :]

    print('Original log_dataset', train.shape)

    # reshape into X=t and Y=t+1
    trainX, trainY = create_lookback_dataset(train, look_back)
    testX, testY = create_lookback_dataset(test, look_back)
    validateX, validateY = create_lookback_dataset(validate, look_back)

    print(trainX.shape, trainY.shape)

    # reshape input to be [samples, time steps, features]
    # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # validateX = np.reshape(validateX, (validateX.shape[0], 1, validateX.shape[1]))



    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(20, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(n_features))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=200, batch_size=20, validation_data=(testX, testY), verbose=2)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    validatePredict = model.predict(validateX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    validatePredict = scaler.inverse_transform(validatePredict)
    validateY = scaler.inverse_transform(validateY)

    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('Entropy loss')
    plt.legend()
    plt.show()

    # calculate root mean squared
    for i in range(len(names)):
        trainRMSE = math.sqrt(mean_squared_error(trainY[:, i], trainPredict[:, i]))
        print('Train Percentage of relative error: {:.2f}'.format(trainRMSE))
        testRMSE = math.sqrt(mean_squared_error(testY[:, i], testPredict[:, i]))
        print('Test Percentage of relative error: {:.2f}'.format(testRMSE))
        validateRMSE = math.sqrt(mean_squared_error(validateY[:, i], validatePredict[:, i]))
        print('Test Percentage of relative error: {:.2f}'.format(validateRMSE))

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset[:, i])
        trainPredictPlot[:] = np.nan
        trainPredictPlot[look_back:len(trainPredict) + look_back] = trainPredict[:, i]

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset[:, i])
        testPredictPlot[:] = np.nan
        testPredictPlot[
        len(trainPredict) + (look_back * 2) + 1:len(dataset) - len(validatePredict) - look_back - 2] = testPredict[:, i]

        # shift Validation predictions for plotting
        validatePredictPlot = np.empty_like(dataset[:, i])
        validatePredictPlot[:] = np.nan
        validatePredictPlot[-len(validatePredict):] = validatePredict[:, i]

        # plot baseline and predictions
        plt.figure()
        plt.plot(scaler.inverse_transform(dataset)[:, i], label='Original')
        plt.plot(trainPredictPlot, label='Train Forecast')
        plt.plot(testPredictPlot, label='Test Forecast')
        plt.plot(validatePredictPlot, label='Validation Forecast')
        plt.title('{}\nTrain Error : {:.3f}\nTest Error : {:.3f}'.format(names[i], trainRMSE, testRMSE))

    plt.show()


# convert an array of values into a log_dataset matrix
def create_lookback_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.asarray(dataX), np.asarray(dataY)


if __name__ == "__main__":
    main()
