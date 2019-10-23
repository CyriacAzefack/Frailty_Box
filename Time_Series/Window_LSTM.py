import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

sns.set_style('darkgrid')
np.random.seed(7)


def main():
    activity = 'meal_preparation'
    raw_dataset = pd.read_csv('./data/{}_dataset.csv'.format(activity))

    # normalize the log_dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(raw_dataset)

    nb_feat = len(raw_dataset.columns)
    dataset = dataset.reshape((len(raw_dataset), nb_feat, 1))

    train_ratio = 0.90
    nb_train = int(train_ratio * len(dataset))
    train, test = dataset[:nb_train], dataset[nb_train:]

    print('Train shape : ', train.shape)
    print('Test shape : ', test.shape)

    nb_input = 10 * nb_feat
    nb_output = 1 * nb_feat

    score, scores, predictions = evaluate_model(train, test, n_input=nb_input, n_output=nb_output, display=True)


    train = train.reshape(train.shape[0], nb_feat)
    test = test.reshape(test.shape[0], nb_feat)
    predictions = predictions.reshape(test.shape[0], nb_feat)

    train_real = scaler.inverse_transform(train)
    test_real = scaler.inverse_transform(test)
    predictions_real = scaler.inverse_transform(predictions)

    train_lin = np.arange(len(train.flatten()))
    test_lin = np.arange(len(train.flatten()), len(train.flatten()) + len(test.flatten()))

    plt.plot(train_lin, train_real.flatten(), c='blue', label='Train')
    plt.plot(test_lin, test_real.flatten(), c='green', label='Test')
    plt.plot(test_lin, predictions_real.flatten(), c='red', label='Forecast')
    plt.legend()
    plt.show()

    summarize_scores('LTSM', score, scores)

    # feat, target = to_supervised(train, n_input=nb_input, n_out=nb_output)
    #
    # print(feat.shape)
    # print(target.shape)


def forecast(model, history, n_input):
    """
    Make a forecast
    :param model:
    :param history:
    :param n_input:
    :return:
    """
    # flatten data
    data = np.asarray(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


def build_model(train, test, n_input, n_output, display=False):
    """
    Training of the model
    :param train:
    :param n_input:
    :return:
    """
    # prepare data
    train_x, train_y = to_supervised(train, n_input, n_output)
    test_x, test_y = to_supervised(test, n_input, n_output)

    print('Features Training : ', train_x.shape)
    print('Target Training : ', train_y.shape)
    # define parameters
    verbose, epochs, batch_size = 2, 50, len(train)
    n_inputs, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(LSTM(10, activation='relu', return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(10, activation='relu', return_sequences=True))
    model.add(LSTM(10, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    # model.add(Dense(50, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                        validation_data=(test_x, test_y))

    if display:
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.title('Entropy loss')
        plt.legend()
        plt.show()
    return model


def to_supervised(train, n_input, n_out):
    """
    Convert history into inputs and outputs
    :param train:
    :param n_input:
    :param n_out:
    :return:
    """
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = [], []
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return np.asarray(X), np.asarray(y)


def summarize_scores(name, score, scores):
    """
    Summarize the scores
    :param name:
    :param score:
    :param scores:
    :return:
    """
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


def evaluate_model(train, test, n_input, n_output, display=False):
    """
    Evaluate a single model
    :param train:
    :param test:
    :param n_input:
    :return:
    """

    # fit model
    model = build_model(train, test, n_input, n_output, display=display)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = np.asarray(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)

    return score, scores, predictions


def evaluate_forecasts(actual, predicted):
    """
    Evaluate one or more weekly forecasts against expected values
    :param actual:
    :param predicted:
    :return:
    """
    scores = list()
    # calculate an RMSE score for each time window
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = math.sqrt(mse)
        # store
        scores.append(rmse)

    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))

    return score, scores


if __name__ == "__main__":
    main()
