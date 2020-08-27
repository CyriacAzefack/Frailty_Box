import datetime as dt
import errno
import math
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pmdarima.arima.utils import ndiffs, nsdiffs
from scipy.stats import expon
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.stattools import acf, pacf

# from Graph_Model.Pattern2Graph import *
from Pattern_Mining.Candidate_Study import modulo_datetime
from Pattern_Mining.Extract_Macro_Activities import compute_episode_occurrences
from Pattern_Mining.Pattern_Discovery import pick_dataset

sns.set_style("darkgrid")


# np.random.seed(1996)

def main():
    dataset_name = 'aruba'
    # dataset_name = 'hh101'
    dataset = pick_dataset(dataset_name)
    # SIM_MODEL PARAMETERS
    episode = ('toilet', 'dress')
    # episode = ('relax',)
    # episode = ('enter_home', 'leave_home', 'watch_tv')
    period = dt.timedelta(days=1)
    time_step = dt.timedelta(minutes=60)
    tep = 30

    ############################
    # PREDICTION PARAMETERS
    train_ratio = 0.8

    # TIME WINDOW PARAMETERS
    window_size = 30
    time_window_duration = dt.timedelta(days=window_size)
    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime() - time_window_duration

    nb_days = int((end_date - start_date) / period) + 1

    activity = MacroActivity(episode=episode, period=period, time_step=time_step, tep=tep)

    tw_id = 0
    while True:
        window_start_date = start_date + tw_id * period
        window_end_date = window_start_date + time_window_duration

        if window_start_date >= end_date:
            break

        window_dataset = dataset[(dataset.date >= window_start_date) & (dataset.date < window_end_date)].copy()

        occurrences, events = compute_episode_occurrences(dataset=window_dataset, episode=episode, tep=tep)

        if len(occurrences) == 0:
            print('No occurrences found !!')
            tw_id += 1
            continue
        # print('{} occurrences of the episode {}'.format(len(occurrences), episode))
        activity.add_time_window(occurrences=occurrences, events=events, time_window_id=tw_id, display=False)

        tw_id += 1

        sys.stdout.write("\r{}/{} Time Windows CAPTURED".format(tw_id, nb_days))
        sys.stdout.flush()
    sys.stdout.write("\n")

    # Dump Parameters
    output = '../output/{}/Activity_{}/'.format(dataset_name, episode)
    # Create the folder if it does not exist yet
    if not os.path.exists(os.path.dirname(output)):
        try:
            os.makedirs(os.path.dirname(output))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # Dump Activity Daily Profile
    pickle.dump(activity.count_histogram, open(output + "/daily_profile.pkl", 'wb'))

    # Dump Activity Durations
    pickle.dump(activity.duration_distrib, open(output + "/durations_distrib.pkl", 'wb'))

    # Dump Execution order
    pickle.dump(activity.occurrence_order, open(output + "/execution_order.pkl", 'wb'))

    # Dump INTER-EVENTS duration
    pickle.dump(activity.expon_lambda, open(output + "/inter_events.pkl", 'wb'))

    activity.plot_time_series()

    # Build the model with a lot of time windows
    print("#####################################")
    print("#    TRAINING FORECASTING MODEL     #")
    print("#####################################")
    error = activity.forecast_history_count(train_ratio=train_ratio, last_time_window_id=tw_id - 1,
                                            nb_periods_to_forecast=10, display=True)

    mean_duration_error, std_duration_error = activity.forecast_durations(train_ratio=train_ratio,
                                                                          window_size=window_size,
                                                                          last_time_window_id=tw_id - 1,
                                                                          nb_periods_to_forecast=10, display=True)


class MacroActivity:
    ID = 0  # Identifier of the macro-activity

    LSTM = 0
    SARIMAX = 1

    def __init__(self, episode, period, time_step, tep=30):
        """
        Creation of a Macro-Activity

        """

        print('\n')
        print("##################################################")
        print("## Creation of the Macro-Activity '{}' ##".format(episode))
        print("####################################################")
        print('\n')
        self.episode = episode
        self.period = period
        self.time_step = time_step
        self.period_ts_index = np.arange(int(period.total_seconds() / time_step.total_seconds()))
        self.tep = dt.timedelta(minutes=tep)

        # ACTIVITY DAILY PROFILE
        # Initialize the histogram for the activity periodicity profile
        hist_columns = ['tw_id'] + ['ts_{}'.format(ts_id) for ts_id in self.period_ts_index]
        self.count_histogram = pd.DataFrame(columns=hist_columns)  # For daily profiles

        # ACTIVITY DURATION PROBABILITY DISTRIBUTIONS
        # Key: label, Value: DataFrame with ['tw_id', 'mean', 'std']
        self.duration_distrib = {}  # For activity duration laws

        for label in self.episode:
            # Stored as Gaussian Distribution
            self.duration_distrib[label] = pd.DataFrame(columns=['tw_id', 'mean', 'std'])

        # EXECUTION ORDER
        self.occurrence_order = pd.DataFrame(
            columns=['tw_id'])  # For execution orders, the columns of the df are like '0132'

        # INTER-EVENTS DURATIONS
        self.expon_lambda = pd.DataFrame(columns=['tw_id', 'lambda'])

        # self.add_time_window(occurrences, events, start_time_window_id, display=display)

        # self.build_histogram(occurrences, display=display)

        MacroActivity.ID += 1

    def __repr__(self):

        str = ''
        if len(self.episode) > 1:
            str += 'MACRO-ACTIVITY'
        else:
            str += 'SINGLE-ACTIVITY'
        str += '{' + ' -- '.join(self.episode) + '}'
        return str

    def get_set_episode(self):
        return frozenset(self.episode)

    def preprocessing(self, occurrences, events):
        '''
        Preprocessing the occurrences and the events
        :param occurrences:
        :return:
        '''
        occurrences['relative_date'] = occurrences.date.apply(
            lambda x: modulo_datetime(x.to_pydatetime(), self.period))
        occurrences['time_step_id'] = occurrences['relative_date'] / self.time_step.total_seconds()
        occurrences['time_step_id'] = occurrences['time_step_id'].apply(math.floor)
        occurrences['activity_duration'] = occurrences.end_date - occurrences.date
        occurrences['activity_duration'] = occurrences['activity_duration'].apply(lambda x: x.total_seconds())

        events['activity_duration'] = events.end_date - events.date
        events['activity_duration'] = events['activity_duration'].apply(lambda x: x.total_seconds())

        return occurrences, events

    def build_histogram(self, occurrences, display=False):
        '''
        Build the Time distribution count_histogram on occurrences
        :param occurrences:
        :param display:
        :return:
        '''

        # Remove duplicates 'day_date - time_step_id'
        occurrences['day_date'] = occurrences.date.apply(lambda x: x.date())
        occurrences.drop_duplicates(['day_date', 'time_step_id'], inplace=True)

        hist = occurrences.groupby(['time_step_id']).count()['date']

        # Create an period_ts_index to have every time steps in the period

        hist = hist.reindex(self.period_ts_index)

        hist.fillna(0, inplace=True)

        # hist = hist/len(hist)  # normalize

        if display:
            plt.bar(hist.index, hist.values)
            plt.title("Activity Daily Profile\n" +
                      '--'.join(self.episode) +
                      '\nTime step : {} min'.format(round(self.time_step.total_seconds() / 60, 1)))
            plt.ylabel('Probability')
            plt.show()

        return hist

    def add_time_window(self, occurrences, events, time_window_id, display=False):
        """
        Update the ocurrence time Histogram and the duration laws history with the new time window data
        :param display:
        :param time_window_id:
        :param events:
        :param occurrences:
        :return:
        """

        occurrences, events = self.preprocessing(occurrences, events)

        if len(self.count_histogram) > 0:
            last_filled_tw_id = self.count_histogram.tw_id.max()
        else:
            last_filled_tw_id = -1

        # Fill the missing time windows data
        hist = self.build_histogram(occurrences, display=display)
        for tw_id in range(last_filled_tw_id + 1, time_window_id):
            self.count_histogram.at[tw_id] = [tw_id] + list(np.zeros(len(hist)))

            for label in self.episode:
                duration_df = self.duration_distrib[label]
                previous_mean_duration, previous_std_duration = 0, 0
                if len(duration_df) > 0:
                    previous_mean_duration, previous_std_duration = duration_df.iloc[-1]['mean'], duration_df.iloc[-1][
                        'std']
                duration_df.at[tw_id] = [tw_id, previous_mean_duration, previous_std_duration]

            self.occurrence_order.at[tw_id, 'tw_id'] = tw_id
            self.expon_lambda.at[tw_id] = [tw_id, 0]

        ## ACTIVITY DAILY PROFILE & ACTIVITIES DURATIONS
        # Update histogram count history
        self.count_histogram.at[time_window_id] = [time_window_id] + list(hist.values.T)

        # print('Histogram count [UPDATED]')
        #
        # UPDATE DURATIONS LAWS
        for label in self.episode:
            label_df = events[events.label == label]
            duration_df = self.duration_distrib[label]

            mean_duration = np.mean(label_df.activity_duration)
            std_duration = np.std(label_df.activity_duration)
            duration_df.at[time_window_id] = [time_window_id, mean_duration, std_duration]

        # print('Duration Gaussian Distribution [UPDATED]')

        ## STOP HERE IF SINGLE ACTIVITY
        if len(self.episode) < 2:
            return

        ## UPDATE EXECUTION ORDER
        occ_order_probability_dict = {}

        # Replace labels by their alphabetic identifier
        for label in self.episode:
            events.loc[events.label == label, 'alph_id'] = sorted(self.episode).index(label)

        for id, occ_row in occurrences.iterrows():
            start_date = occ_row.date
            end_date = start_date + self.tep
            arr = list(events.loc[(events.date >= start_date) & (events.date < end_date), 'alph_id'].values)
            # remove duplicates
            arr = list(dict.fromkeys(arr))
            occ_order_str = ''.join([str(int(x)) for x in arr])

            if occ_order_str in occ_order_probability_dict:
                occ_order_probability_dict[occ_order_str] += 1 / len(occurrences)
            else:
                occ_order_probability_dict[occ_order_str] = 1 / len(occurrences)

        self.occurrence_order.at[time_window_id, 'tw_id'] = time_window_id
        for occ_order_str, prob in occ_order_probability_dict.items():
            self.occurrence_order.at[time_window_id, occ_order_str] = prob

        self.occurrence_order.fillna(0, inplace=True)

        # print('Execution Order Probability [UPDATED]')

        ## UPDATE INTER-EVENTS DURATION
        # Add occ_id to events
        events["occ_id"] = events.index
        events["occ_id"] = events.occ_id.apply(lambda x: math.floor(x / len(self.episode)))

        events = events.join(events.groupby(['occ_id'])[['date']].shift(-1).add_suffix('_next'))
        events.rename(columns={events.columns[-1]: "date_next"}, inplace=True)

        events.dropna(inplace=True)
        inter_event_durations = (events.date_next - events.date).apply(lambda x: x.total_seconds()).values

        # Fit an exponential distribution
        loc, scale = expon.fit(inter_event_durations.astype(np.float64), floc=0)

        self.expon_lambda.at[time_window_id] = [time_window_id, 1 / scale]

    def get_count_histogram(self, time_window_id):
        """
        :param time_window_id: time window identifier
        :return: the histogram count at a specific time window
        """
        return self.count_histogram.loc[[time_window_id]]

    def forecast_history_count(self, train_ratio, last_time_window_id, nb_periods_to_forecast,
                               display=False):
        """
        Fit a time series forecasting model to the history count data
        :param train_ratio:
        :param last_time_window_id: Last time window id registered by the manager
        :param nb_periods_to_forecast:
        :param display:
        :return: Normalised Mean Squared Error (NMSE)
        """

        nb_tstep = len(self.count_histogram.columns) - 1
        nb_steps_to_forecast = nb_periods_to_forecast * nb_tstep

        # Fill history count df until last time window registered
        last_filled_tw_id = int(self.count_histogram.tw_id.max())

        for tw_id in range(last_filled_tw_id + 1, last_time_window_id):
            self.count_histogram.at[tw_id] = [tw_id] + list(np.zeros(nb_tstep))

        raw_dataset = self.count_histogram.drop(['tw_id'], axis=1)
        dataset = raw_dataset.values.flatten()

        if len(dataset) < 10:  # Can't train on such less data
            return None

        dataset = dataset.astype(int)

        train_size = int(len(dataset) * train_ratio)

        train, test = dataset[:train_size], dataset[train_size:]

        test_size = test.shape[0]

        sarima_model = sarimax.SARIMAX(train, order=(1, 0, 0), seasonal_order=(1, 0, 1, nb_tstep),
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)

        sarima_model = sarima_model.fit(disp=False)

        raw_forecast = sarima_model.predict(start=train_size, end=-1 + train_size + test_size + nb_steps_to_forecast)

        raw_forecast = pd.Series(raw_forecast)

        validation_forecast = raw_forecast[:test_size]
        mse_error = mean_squared_error(test, validation_forecast)

        # Forecast to use
        forecasts = raw_forecast[test_size:]

        # Replace all negative values by 0
        forecasts = np.where(forecasts > 0, forecasts, 0)
        forecasts = forecasts.reshape((nb_periods_to_forecast, nb_tstep))

        current_tw_id = last_time_window_id
        for hist_count in forecasts:
            current_tw_id += 1
            self.count_histogram.at[current_tw_id] = [current_tw_id] + list(hist_count)

        if display:
            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(train_size, train_size + len(validation_forecast)), validation_forecast, 'r')
            plt.plot(dataset, 'b', alpha=0.5)
            plt.plot(np.arange(len(dataset), len(dataset) + len(raw_forecast)), raw_forecast, 'red', alpha=0.7)

            plt.title('{}\nTest MSE: {:.3f}'.format(self.episode, mse_error))
            plt.xlabel('Time')
            plt.ylabel('count')
            plt.axvline(x=train_size, color='black')
            plt.show()

        return mse_error

    def forecast_durations(self, train_ratio, last_time_window_id, nb_periods_to_forecast, window_size, display=False):
        """
        forecast the activities duration
        :param train_ratio:
        :param last_time_window_id:
        :param nb_periods_to_forecast:
        :param display:
        :return: Two dict like object {label: r2_score}
        """
        mean_duration_errors = {}
        std_duration_errors = {}

        for label in self.episode:
            print('[{}] Duration forecasting...'.format(label))

            # Fill history count df until last time window registered

            last_filled_tw_id = int(self.duration_distrib[label].tw_id.max())

            for tw_id in range(last_filled_tw_id + 1, last_time_window_id):
                self.duration_distrib[label].at[tw_id] = [tw_id, 0, 0]

            forecast_df = pd.DataFrame(columns=['tw_id', 'mean', 'std'])

            forecast_df.tw_id = np.arange(last_time_window_id, last_time_window_id + nb_periods_to_forecast + 1)

            # Fit & Predict Duration Mean values
            print('Mean Duration Forecasting...')
            mean_duration_data = list(self.duration_distrib[label]['mean'].values)

            if len(mean_duration_data) * train_ratio < 2:
                forecast_df['mean'] = [0] * nb_periods_to_forecast
                forecast_df['std'] = [0] * nb_periods_to_forecast
                self.duration_distrib[label] = self.duration_distrib[label].append(forecast_df, ignore_index=True)
                return [np.nan], [np.nan]

            mean_duration_error, mean_duration_forecast_values = arima_forecast(data=mean_duration_data,
                                                                                train_ratio=train_ratio,
                                                                                seasonality=window_size,
                                                                                nb_steps_to_forecast=nb_periods_to_forecast,
                                                                                label=f'{label} - Mean Duration',
                                                                                display=display)
            print("Mean Duration Error (NMSE) : {:.2f}".format(mean_duration_error))
            mean_duration_errors[label] = mean_duration_error

            # Fit & Predict Duration STD values
            print('STD Duration Forecasting...')
            std_duration_data = list(self.duration_distrib[label]['std'].values)

            std_duration_error, std_duration_forecast_values = arima_forecast(data=std_duration_data,
                                                                              train_ratio=train_ratio,
                                                                              seasonality=window_size,
                                                                              nb_steps_to_forecast=nb_periods_to_forecast,
                                                                              label=f'{label} - STD Duration',
                                                                              display=display)

            print("STD Duration Error (NMSE) : {:.2f}".format(std_duration_error))
            std_duration_errors[label] = std_duration_error



            forecast_df['mean'] = mean_duration_forecast_values
            forecast_df['std'] = std_duration_forecast_values

            self.duration_distrib[label] = self.duration_distrib[label].append(forecast_df, ignore_index=True)

            if display:
                len_available_data = len(mean_duration_data)

                plt.figure(figsize=(15, 5))
                plt.subplot(121)
                plt.plot(np.asarray(mean_duration_data) / 60, color='b', alpha=0.6)
                plt.plot(np.arange(len_available_data, len_available_data + nb_periods_to_forecast),
                         mean_duration_forecast_values / 60, color='red')
                plt.xlabel('Timepoints')
                plt.ylabel('Mean Duration (mn)')

                plt.subplot(122)
                plt.plot(np.asarray(std_duration_data) / 60, color='b', alpha=0.6)
                plt.plot(np.arange(len_available_data, len_available_data + nb_periods_to_forecast),
                         std_duration_forecast_values / 60, color='red')
                plt.xlabel('Timepoints')
                plt.ylabel('Std Duration (mn)')

                plt.title('Duration of Activity \'{}\''.format(label))

                plt.tight_layout()
                plt.show()

        return mean_duration_errors, std_duration_errors

    def forecast_execution_order(self, train_ratio, last_time_window_id, nb_periods_to_forecast,
                                 display=False):
        """
        Fit the execution order
        :param train_ratio:
        :param last_time_window_id:
        :param nb_periods_to_forecast:
        :param display:
        :return:
        """

        pass

    def simulate(self, start_date, time_step_id, time_window_id):
        """
        Generate events on the macro-activity
        :param start_date:
        :param time_step_id:
        :param time_window_id:
        :return:
        """

        events = pd.DataFrame(columns=['date', 'end_date', 'label'])

        current_date = start_date

        # TODO : Take the execution order into account
        for label in self.episode:
            mean = self.duration_distrib[label].loc[time_window_id]['mean']
            std = self.duration_distrib[label].loc[time_window_id]['std']

            # To avoid negative durations
            if (mean == 0) and (std == 0):
                continue
            duration = -1
            while duration < 0:
                duration = math.ceil(np.random.normal(mean, std))

            end_date = current_date + dt.timedelta(seconds=duration)
            events.loc[len(events)] = [current_date, end_date, label]

            current_date = end_date

        return events

    def plot_time_series(self):
        """
        Display all the time series present in the Macro-Activity
        :return:
        """

        ## ACTIVITY DAILY PROFILE Time Series
        ######################################

        raw_dataset = self.count_histogram.drop(['tw_id'], axis=1)
        dataset = raw_dataset.values.flatten()

        plt.figure()
        plt.plot(dataset)
        plt.title("Histogram Count")

        ## ACTIVITY DURATIONS
        #####################
        plt.figure()
        for label in self.episode:
            df = self.duration_distrib[label]
            df['mean'] /= 60
            plt.plot(df.tw_id, df['mean'], label=label)
            # sns.lineplot(x='tw_id', y='mean', data=df, label=label)
        plt.title('Mean Activity Duration')
        plt.xlabel('Time Windows ID')
        plt.ylabel('Duration (min)')
        plt.legend()

        # EXECUTION ORDER

        if len(self.episode) < 2:
            plt.show()
            return

        df = self.occurrence_order
        df = df.melt('tw_id', var_name='cols', value_name='vals')
        g = sns.factorplot(x="tw_id", y="vals", hue='cols', data=df)

        plt.xlabel('Time Windows ID')
        plt.ylabel('Probability')
        plt.title("Execution Order : {}".format(self.episode))
        plt.legend()

        # INTER-EVENTS DURATIONS
        plt.figure()
        df = self.expon_lambda
        plt.plot(df['tw_id'], df['lambda'])
        # sns.lineplot(x='tw_id', y='lambda', data=df)

        plt.xlabel('Time Windows ID')
        plt.ylabel('Lambda')
        plt.title('Exponenital Distrib Parameter')

        plt.show()

    def dump_data(self, output):
        """
        Dump All the data in the output_dir
        :return:
        """

        # Create the directory for the macro-activity

        output += f'{str(self)}/'
        # Create the folder if it does not exist yet
        if not os.path.exists(os.path.dirname(output)):
            try:
                os.makedirs(os.path.dirname(output))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # Dump the histogram count
        self.count_histogram.to_csv(output + "/count_histogram.csv", index=False, sep=";")

        for label, duration_distrib in self.duration_distrib.items():
            # Stored as Gaussian Distribution
            duration_distrib.to_csv(output + f"{label}_duration_distribution.csv", index=False, sep=";")


def fit_and_forecast(data, train_ratio, nb_periods_predict):
    """
    Fit and predict the values of the time series
    :param data:
    :return:
    """

    train_size = int(train_ratio * len(data))

    train, test = data[:train_size], data[train_size:]

    # plt.plot(train)
    # plt.show()

    model = ARIMA(train, order=(2, 1, 0))
    # model = SARIMAX(train, order=(4, 0, 0), seasonal_order=(0, 0, 0, 1), enforce_stationarity=True,
    #         enforce_invertibility=False)

    model_fit = model.fit(disp=False)

    validation_forecast = model_fit.forecast(len(test))[0]

    error = mean_squared_error(test, validation_forecast) / np.mean(test)

    real_forecast = model_fit.forecast(len(test) + nb_periods_predict)[0][len(test):]

    return error, real_forecast


def arima_forecast(data, train_ratio, seasonality, nb_steps_to_forecast, label, display=False):
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

    # sarima_model = pm.auto_arima(train,
    #                              start_p=0, max_p=1,
    #                              start_q=0, max_q=1,
    #                              start_P=0, max_P=1,
    #                              start_Q=0, max_Q=1, max_order=2,
    #                              m=seasonality, seasonal=True,
    #                              d=d,
    #                              max_iter=50,
    #                              method='lbfgs',
    #                              trace=False,
    #                              error_action='ignore',  # don't want to know if an order does not work
    #                              suppress_warnings=True,  # don't want convergence warnings
    #                              stepwise=True)  # set to stepwise

    # raw_forecast = sarima_model.predict(test_size + nb_steps_to_forecast)

    # p, d, q = sarima_model.order
    # P, D, Q, m = sarima_model.seasonal_order
    p, d, q = 0, d, 1
    P, D, Q, m = 0, 0, 0, 0

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
    forecasts = raw_forecast[test_size:].values

    # Replace all negative values by 0
    forecasts = np.where(forecasts > 0, forecasts, 0)
    # forecasts = forecasts.reshape((nb_periods_to_forecast, nb_tstep))

    if display:
        # Visualize the forecasts (blue=train, green=forecasts)
        # print(sarima_model.summary())
        plt.plot(train, label='Training data')
        plt.plot(test, label='Test data')
        # plt.plot(np.arange(train_size, train_size + len(raw_forecast)), raw_forecast, label='Predicted data')
        plt.plot(all_forecast, label='Predicted data')
        plt.title(f'{label}\nMSE ERROR : {nmse_error:.3f}')
        plt.legend()
        plt.show()

    return nmse_error, forecasts


def find_ARIMA_params(data, seasonality):
    """

    :param data:
    :param seasonality:
    :return:
    """

    plt.plot(data)
    plt.show()

    diff_train = pd.Series(np.log(data))

    diff_train = diff_train.diff(periods=seasonality)[seasonality:]
    diff_train.replace([np.inf, -np.inf], np.nan, inplace=True)

    diff_train = diff_train.fillna(0)

    #
    plt.plot(diff_train)
    plt.show()
    #
    lag_acf = acf(diff_train, nlags=20)
    lag_pacf = pacf(diff_train, nlags=20, method='ols')

    born_supp = 1.96 / np.sqrt(len(diff_train))
    born_inf = -born_supp

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


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


if __name__ == '__main__':
    main()
