# import seaborn as sns
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

from Graph_Model.Pattern2Graph import *
from Pattern_Mining.Candidate_Study import modulo_datetime
from Pattern_Mining.Extract_Macro_Activities import compute_episode_occurrences
from Pattern_Mining.Pattern_Discovery import pick_dataset


# sns.set_style('darkgrid')


# np.random.seed(1996)

def main():
    dataset = pick_dataset('HH101')
    # SIM_MODEL PARAMETERS
    episode = ('sleep',)
    period = dt.timedelta(days=1)
    time_step = dt.timedelta(minutes=60)
    tep = 30

    # PREDICTION PARAMETERS
    train_ratio = 0.8

    # TIME WINDOW PARAMETERS
    time_window_duration = dt.timedelta(days=21)
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

        # print('{} occurrences of the episode {}'.format(len(occurrences), episode))
        activity.add_time_window(occurrences=occurrences, events=events, time_window_id=tw_id, display=False)


        tw_id += 1

        sys.stdout.write("\r{}/{} Time Windows CAPTURED".format(tw_id, nb_days))
        sys.stdout.flush()
    sys.stdout.write("\n")

    # Build the model with a lot of time windows
    print("#####################################")
    print("#    FORECASTING MODEL TRAINING     #")
    print("#####################################")
    # error = activity.fit_history_count_forecasting_model(train_ratio=train_ratio, last_time_window_id=tw_id-1,
    #                                                      nb_periods_to_forecast=30, display=True)

    mean_duration_error, std_duration_error = activity.fit_duration_distrub_forecasting_model(train_ratio=train_ratio,
                                                                                              last_time_window_id=tw_id - 1,
                                                                                              nb_periods_to_forecast=10,
                                                                                              display=True)

    # print("Prediction Error (NMSE) : {:.2f}".format(error))




class MacroActivity:
    ID = 0  # Identifier of the macro-activity

    LSTM = 0
    SARIMAX = 1

    def __init__(self, episode, period, time_step, tep=30):
        '''
        Creation of a Macro-Activity
        :param episode:
        :param occurrences: dataset of occurrences of the episode
        :param events: sublog of events from the input event log
        :param period: periodicity of the analysis
        :param time_step: Time step for the histogram (used for the prediction)
        :param start_time_window_id:
        :param tep:
        :param display:
        '''

        print('\n')
        print("##################################################")
        print("## Creation of the Macro-Activity '{}' ##".format(episode))
        print("####################################################")
        print('\n')
        self.episode = episode
        self.period = period
        self.time_step = time_step
        self.index = np.arange(int(period.total_seconds() / time_step.total_seconds()))
        self.tep = dt.timedelta(minutes=tep)

        # ACTIVITY DAILY PROFILE
        # Initialize the histogram for the activity periodicity profile
        hist_columns = ['tw_id'] + ['ts_{}'.format(ts_id) for ts_id in self.index]
        self.count_histogram = pd.DataFrame(columns=hist_columns)  # For daily profiles

        # ACTIVITY DURATION PROBABILITY DISTRIBUTIONS
        # Key: label, Value: DataFrame with ['tw_id', 'mean', 'std']
        self.duration_distrib = {}  # For activity duration laws

        for label in self.episode:
            # Stored as Gaussian Distribution
            self.duration_distrib[label] = pd.DataFrame(columns=['tw_id', 'mean', 'std'])

        # TREE GRAPH TRANSITION MATRIX
        self.occurrence_order = pd.DataFrame(
            columns=['tw_id'])  # For execution orders, the columns of the df are like '0132'

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

        hist = occurrences.groupby(['time_step_id']).count()['date']

        # Create an index to have every time steps in the period

        hist = hist.reindex(self.index)

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
        :param occurrences:
        :return:
        """
        occurrences, events = self.preprocessing(occurrences, events)

        # Update histogram count history
        hist = self.build_histogram(occurrences, display=display)

        if len(self.count_histogram) > 0:
            last_filled_tw_id = self.count_histogram.tw_id.max()
        else:
            last_filled_tw_id = -1

        # Fill the missing time windows data

        for tw_id in range(last_filled_tw_id + 1, time_window_id):
            self.count_histogram.at[tw_id] = [tw_id] + list(np.zeros(len(hist)))

            for label in self.episode:
                duration_df = self.duration_distrib[label]
                duration_df.at[tw_id] = [tw_id, 0, 0]

            self.occurrence_order.at[tw_id, 'tw_id'] = tw_id



        self.count_histogram.at[time_window_id] = [time_window_id] + list(hist.values.T)

        # print('Histogram count [UPDATED]')
        #
        # Update duration laws
        for label in self.episode:
            label_df = events[events.label == label]
            duration_df = self.duration_distrib[label]

            mean_duration = np.mean(label_df.activity_duration)
            std_duration = np.std(label_df.activity_duration)
            duration_df.at[time_window_id] = [time_window_id, mean_duration, std_duration]

        # print('Duration Gaussian Distribution [UPDATED]')

        # Update execution order

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

    def get_count_histogram(self, time_window_id):
        """
        :param time_window_id: time window identifier
        :return: the histogram count at a specific time window
        """
        return self.count_histogram.loc[[time_window_id]]

    def fit_history_count_forecasting_model(self, train_ratio, last_time_window_id, nb_periods_to_forecast,
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

        # Fill history count df unti last time window registered

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


        monitoring_start_time = t.time()

        # find_ARIMA_params(train, seasonality=nb_tstep)
        # if method == MacroActivity.SARIMAX:
        model = SARIMAX(train, order=(2, 0, 0), seasonal_order=(2, 1, 0, nb_tstep), enforce_stationarity=False,
                        enforce_invertibility=False)

        if np.sum(model.start_params) == 0:  # We switch to a simple ARIMA model
            print('Simple ARIMA Model')
            model = ARIMA(train, order=(4, 1, 4))
            # print(mode)
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(len(test))[0]
        else:

            # burnin_model = SARIMAX(train, order=(4, 1, 4), seasonal_order=(1, 0, 0, 1), enforce_stationarity=False,
            #                        enforce_invertibility=False)
            # start_params = burnin_model.fit(return_params=True, disp=False)
            #
            # start_params = np.asarray(list(start_params) + [0]*(len(model.start_params) - len(start_params)))

            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(len(test))

        elapsed_time = dt.timedelta(seconds=round(t.time() - monitoring_start_time, 1))

        print("Training Time: {}".format(elapsed_time))

        error = mean_squared_error(test, forecast) / np.mean(test)


        # self.hist_count_forecasting_model = model_fit

        # Fill the forecasting

        nb_steps_to_forecast = len(test) + nb_periods_to_forecast * nb_tstep
        raw_forecast = model_fit.forecast(nb_steps_to_forecast)[len(test):]

        # Replace all negative values by 0
        raw_forecast = np.where(raw_forecast > 0, raw_forecast, 0)
        real_forecast = raw_forecast.reshape((nb_periods_to_forecast, nb_tstep))

        current_tw_id = last_time_window_id
        for hist_count in real_forecast:
            current_tw_id += 1
            self.count_histogram.at[current_tw_id] = [current_tw_id] + list(hist_count)


        if display:
            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(train_size, train_size + len(forecast)), forecast, 'r')
            plt.plot(dataset, 'b', alpha=0.5)
            plt.plot(np.arange(len(dataset), len(dataset) + len(raw_forecast)), raw_forecast, 'red', alpha=0.7)

            plt.title('{}\nTest NMSE: {:.3f}'.format(self.episode, error))
            plt.xlabel('Time')
            plt.ylabel('count')
            plt.axvline(x=train_size, color='black')
            plt.show()

        return error

    def fit_duration_distrub_forecasting_model(self, train_ratio, last_time_window_id, nb_periods_to_forecast,
                                               display=False):
        """
        Fit the
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

            # Fill history count df unti last time window registered

            last_filled_tw_id = int(self.duration_distrib[label].tw_id.max())

            for tw_id in range(last_filled_tw_id + 1, last_time_window_id):
                self.duration_distrib[label].at[tw_id] = [tw_id, 0, 0]

            # Fit & Predict Duration Mean values
            mean_duration_data = list(self.duration_distrib[label]['mean'].values)

            # TODO : Use the real 'start_date'
            start_date = dt.datetime.now().date()
            start_date = dt.datetime.combine(start_date, dt.datetime.min.time())
            mean_duration_error, mean_duration_forecast_values = prophet_forecaster(start_date=start_date,
                                                                                    data=mean_duration_data,
                                                                                    train_ratio=train_ratio,
                                                                                    nb_period_to_forecast=nb_periods_to_forecast,
                                                                                    display=display)
            print("Mean Duration Error (NMSE) : {:.2f}".format(mean_duration_error))
            mean_duration_errors[label] = mean_duration_error

            # Fit & Predict Duration STD values
            std_duration_data = list(self.duration_distrib[label]['std'].values)

            std_duration_error, std_duration_forecast_values = prophet_forecaster(start_date=start_date,
                                                                                  data=std_duration_data,
                                                                                  train_ratio=train_ratio,
                                                                                  nb_period_to_forecast=nb_periods_to_forecast,
                                                                                  display=display)
            print("STD Duration Error (NMSE) : {:.2f}".format(std_duration_error))
            std_duration_errors[label] = std_duration_error

            forecast_df = pd.DataFrame(columns=['tw_id', 'mean', 'std'])

            forecast_df.tw_id = np.arange(last_time_window_id + 1, last_time_window_id + nb_periods_to_forecast + 1)

            forecast_df['mean'] = mean_duration_forecast_values
            forecast_df['std'] = std_duration_forecast_values

            self.duration_distrib[label] = self.duration_distrib[label].append(forecast_df, ignore_index=True)

            if display:
                len_available_data = len(mean_duration_data)
                # Plot ACF:
                plt.figure(figsize=(15, 5))
                plt.subplot(121)
                plt.plot(np.asarray(mean_duration_data) / 60, color='b', alpha=0.6)
                plt.plot(np.arange(len_available_data, len_available_data + nb_periods_to_forecast),
                         mean_duration_forecast_values / 60, color='red')
                plt.xlabel('Timepoints')
                plt.ylabel('Mean Duration (mn)')

                # Plot PACF :
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


def prophet_forecaster(start_date, data, train_ratio, nb_period_to_forecast, display=False):
    """
    Forecast using Prophet from Facebook
    :param start_date:
    :param data:
    :param train_ratio:
    :param nb_period_to_forecast:
    :return:
    """

    period = dt.timedelta(days=1)

    date_range = [start_date + i * period for i in range(len(data))]

    dataset = pd.DataFrame(list(zip(date_range, data)), columns=['ds', 'y'])

    n_train = int(len(dataset) * train_ratio)
    train_dataset = dataset[:n_train]
    test_dataset = dataset[n_train:]

    model = Prophet(interval_width=0.95, daily_seasonality=False)

    with suppress_stdout_stderr():
        model.fit(train_dataset)


    freq = '{}S'.format(int(period.total_seconds()))
    future = model.make_future_dataframe(periods=len(test_dataset) + nb_period_to_forecast, freq=freq)
    forecast = model.predict(future)

    test_indexes = np.arange(n_train, n_train + len(test_dataset))
    validation_forecast = forecast.loc[test_indexes, 'yhat'].values

    test = test_dataset['y'].values

    error = mean_squared_error(test, validation_forecast) / np.mean(test)


    error = round(error, 3)

    real_forecast = forecast.tail(nb_period_to_forecast)['yhat'].values

    # Remove all negative values
    real_forecast = np.where(real_forecast > 0, real_forecast, 0)

    # df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')

    if display:
        ax = sns.lineplot(x="ds", y="y", data=train_dataset, label='Train')
        sns.lineplot(x="ds", y='yhat', data=forecast, label='Forecast', ax=ax)
        sns.lineplot(x="ds", y='y', data=test_dataset, label='Test', ax=ax)

        plt.title('R2 score : {}'.format(error))

        # model.plot(forecast)
        # model.plot_components(forecast)
        plt.show()

    return error, real_forecast


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
