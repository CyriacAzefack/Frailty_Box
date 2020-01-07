import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.models import Sequential
from sklearn.metrics import mean_squared_error

sns.set_style('darkgrid')
np.random.seed(7)


def main():
    dataset_name = "aruba"

    episode = ('bed_to_toilet', 'sleeping')
    # episode = ('toilet', 'dress')

    output = "C:/Users/cyriac.azefack/Workspace/Frailty_Box/output/{}/Activity_{}/".format(dataset_name, episode)

    daily_profile = pickle.load(open(output + "daily_profile.pkl", "rb"))

    duration_distrib = pickle.load(open(output + "durations_distrib.pkl", "rb"))

    occurrence_order = pickle.load(open(output + "execution_order.pkl", "rb"))

    inter_events = pickle.load(open(output + "inter_events.pkl", "rb"))

    features_names = []

    dataset = inter_events
    features_names.append('lambda_inter_events')

    for label in episode:
        data = duration_distrib[label]
        dataset = dataset.join(data.set_index('tw_id'), on='tw_id', rsuffix='_' + label)
        features_names.append('mean_{}'.format(label))
        features_names.append('std_{}'.format(label))

    dataset = dataset.join(occurrence_order.set_index('tw_id'), on='tw_id')
    features_names += list(occurrence_order.columns[1:])

    dataset.drop(columns=['tw_id'], inplace=True)

    dataset.columns = features_names
    n_features = len(features_names)

    dataset = dataset.values
    # log_dataset = log_dataset.values.reshape((len(log_dataset), n_features, 1))
    #
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # log_dataset = scaler.fit_transform(log_dataset)

    print(features_names)
    look_back = 5
    nb_periods = 20
    train_ratio = 0.8
    target_index = 0

    nb_train = int(train_ratio * len(dataset))
    train, test = dataset[:nb_train], dataset[nb_train:]

    nb_input = look_back
    nb_output = nb_periods

    score, scores, predictions = evaluate_model(train, test, n_input=nb_input, n_output=nb_output, display=True,
                                                target_index=target_index)

    summarize_scores('LTSM', score, scores)

    # train = scaler.inverse_transform(train)
    # test = scaler.inverse_transform(test)
    # predictions = scaler.inverse_transform(predictions)

    # plot scores
    days = [i for i in range(nb_output)]
    plt.plot(days, scores, marker='o', label='lstm')
    plt.title('{}\nOverall RMSE:{:.3f}'.format(features_names[target_index], score))
    plt.show()

    big_prediction = []

    for i in range(len(predictions)):
        predict = predictions[i]
        array = np.zeros(len(predictions) + nb_output)
        array[i:i + len(predict)] = predict

        big_prediction.append(array)

    big_prediction = np.asarray(big_prediction)

    real_prediction = np.divide(np.sum(big_prediction, axis=0), np.count_nonzero(big_prediction, axis=0))
    testPredictPlot = np.empty(len(test))
    testPredictPlot[:] = test[:, target_index]

    predictPlot = np.empty(len(test))
    predictPlot[:] = np.nan
    predictPlot[nb_input:nb_input + len(real_prediction)] = real_prediction

    plt.figure()
    plt.plot(testPredictPlot, label='Original Test')
    plt.plot(predictPlot, label='Test Forecast')
    # plt.plot(testPredictPlot, label='Test Forecast')
    # plt.plot(validatePredictPlot, label='Validation Forecast')
    plt.title('{}\nOverall RMSE Error : {:.3f}'.format(features_names[target_index], score))
    plt.legend()
    plt.show()

    pass
    # train = train.reshape(train.shape[0], nb_feat)
    # test = test.reshape(test.shape[0], nb_feat)
    # predictions = predictions.reshape(test.shape[0], nb_feat)
    #
    # train_real = scaler.inverse_transform(train)
    # test_real = scaler.inverse_transform(test)
    # predictions_real = scaler.inverse_transform(predictions)
    #
    # train_lin = np.arange(len(train.flatten()))
    # test_lin = np.arange(len(train.flatten()), len(train.flatten()) + len(test.flatten()))
    #
    # plt.plot(train_lin, train_real.flatten(), c='blue', label='Train')
    # plt.plot(test_lin, test_real.flatten(), c='green', label='Test')
    # plt.plot(test_lin, predictions_real.flatten(), c='red', label='Forecast')
    # plt.legend()
    # plt.show()

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

    # data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, 1]
    # input_x = input_x.reshape((1, n_input, input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


def build_model(train, test, n_input, n_output, target_index, display=False):
    """
    Training of the model
    :param train:
    :param n_input:
    :return:
    """
    # prepare data
    train_x, train_y = to_supervised(train, n_input, n_output, target_index)
    test_x, test_y = to_supervised(test, n_input, n_output, target_index)

    print('Features Training : ', train_x.shape)
    print('Target Training : ', train_y.shape)
    # define parameters
    verbose, epochs, batch_size = 1, 50, 10
    n_inputs, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(LSTM(10, activation='relu', return_sequences=True, input_shape=(n_inputs, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(10, activation='relu', return_sequences=True))
    model.add(LSTM(10, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    # model.add(LSTM(200, activation='relu', input_shape=(n_inputs, n_features)))
    # model.add(TimeDistributed(Dense(100, activation='relu')))
    # model.add(Dense(n_outputs))
    # model.compile(loss='mse', optimizer='adam')
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


def to_supervised(train, n_input, n_out, target_index):
    """
    Convert history into inputs and outputs
    :param train:
    :param n_input:
    :param n_out:
    :return:
    """
    # flatten data
    # data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    data = train
    X, y = [], []
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            x_input = data[in_start:in_end, :]
            # x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, target_index])
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


def evaluate_model(train, test, n_input, n_output, target_index=0, display=False):
    """
    Evaluate a single model
    :param train:
    :param test:
    :param n_input:
    :return:
    """

    train_x, train_y = to_supervised(train, n_input, n_output, target_index)
    test_x, test_y = to_supervised(test, n_input, n_output, target_index)
    # fit model
    model = build_model(train, test, n_input, n_output, target_index, display=display)
    # history is a list of weekly data
    history = [x for x in train_x]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test_x)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test_x[i, :])
    # evaluate predictions days for each week
    predictions = np.asarray(predictions)

    score, scores = evaluate_forecasts(test_y, predictions)

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
