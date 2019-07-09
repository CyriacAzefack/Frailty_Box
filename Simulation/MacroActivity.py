import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from Graph_Model.Pattern2Graph import *
from Pattern_Mining.Candidate_Study import modulo_datetime
from Pattern_Mining.Extract_Macro_Activities import compute_episode_occurrences
from Pattern_Mining.Pattern_Discovery import pick_dataset

sns.set_style('darkgrid')


# np.random.seed(1996)

def main():
    dataset = pick_dataset('aruba', nb_days=50)

    # SIM_MODEL PARAMETERS
    episode = ('relax', 'meal_preparation')
    period = dt.timedelta(days=1)
    time_step = dt.timedelta(minutes=60)
    tep = 30

    # PREDICTION PARAMETERS
    train_ratio = 0.8

    # TIME WINDOW PARAMETERS
    time_window_duration = dt.timedelta(days=30)
    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime() - time_window_duration
    window_start_date = start_date

    nb_days = int((end_date - start_date) / period) + 1

    activity = MacroActivity(episode=episode, period=period, time_step=time_step, tep=tep)


    tw_id = 0
    while window_start_date < end_date:
        window_end_date = window_start_date + time_window_duration

        window_dataset = dataset[(dataset.date >= window_start_date) & (dataset.date < window_end_date)].copy()

        occurrences, events = compute_episode_occurrences(dataset=window_dataset, episode=episode, tep=tep)

        # print('{} occurrences of the episode {}'.format(len(occurrences), episode))
        activity.add_time_window(occurrences=occurrences, events=events, time_window_id=tw_id, display=False)

        x = activity.get_count_histogram(time_window_id=tw_id)
        x.drop(['tw_id'], axis=1, inplace=True)
        x.index = [frozenset(episode)]

        window_start_date += period
        tw_id += 1

        sys.stdout.write("\r{}/{} Time Windows CAPTURED".format(tw_id, nb_days))
        sys.stdout.flush()
    sys.stdout.write("\n")

    # Build the model with a lot of time windows
    print("#####################################")
    print("#    FORECASTING MODEL TRAINING     #")
    print("#####################################")
    error = activity.fit_history_count_forecasting_model(train_ratio=train_ratio, method=MacroActivity.SARIMAX,
                                                         display=True)

    print("Prediction Error (NMSE) : {:.2f}".format(error))




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

        if len(self.count_histogram) > 1:
            max_tw_id = self.count_histogram.tw_id.max()

            # Fill the missing time windows data

            for tw_id in range(max_tw_id + 1, time_window_id):
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

    def fit_history_count_forecasting_model(self, train_ratio, method, display=False):
        """
        Fit a time series forecasting model to the history count data
        :param method:
        :return: Normalised Mean Squared Error (NMSE)
        """
        raw_dataset = self.count_histogram.drop(['tw_id'], axis=1)
        nb_tstep = len(raw_dataset.columns)

        if len(raw_dataset) < 10:  # Can't train such less data
            return -10
        dataset = raw_dataset.values.flatten()

        dataset = dataset.astype(int)

        train_size = int(len(dataset) * train_ratio)

        train, test = dataset[:train_size], dataset[train_size:]

        if method == MacroActivity.SARIMAX:
            model = SARIMAX(train, order=(4, 1, 4), seasonal_order=(1, 0, 0, nb_tstep), enforce_stationarity=False,
                            enforce_invertibility=False)

            if np.sum(model.start_params) == 0:  # We switch to a simple ARIMA model
                model = ARIMA(train, order=(4, 1, 4))
                # print(mode)
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(len(test))[0]
            else:
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(len(test))

        error = mean_squared_error(test, forecast) / np.mean(test)
        if display:
            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(train_size, train_size + len(forecast)), forecast, 'r')
            plt.plot(dataset, 'b')

            plt.title('{}\nTest NMSE: {:.3f}'.format(self.episode, error))
            plt.xlabel('Time')
            plt.ylabel('count')
            plt.axvline(x=train_size, color='black')
            plt.show()

        return error

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

        for label in self.episode:
            mean = self.duration_distrib[label].loc[time_window_id]['mean']
            std = self.duration_distrib[label].loc[time_window_id]['std']

            # To avoid negative durations
            duration = -1
            while duration < 0:
                duration = math.ceil(np.random.normal(mean, std))

            end_date = current_date + dt.timedelta(seconds=duration)
            events.loc[len(events)] = [current_date, end_date, label]

            current_date = end_date

        return events








if __name__ == '__main__':
    main()
