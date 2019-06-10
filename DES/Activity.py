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
    dataset = pick_dataset('hh101')

    episode = ('sleep',)
    period = dt.timedelta(days=1)
    time_step = dt.timedelta(minutes=10)
    tep = 30

    occurrences, events = compute_episode_occurrences(dataset=dataset, episode=episode, tep=tep)

    print('{} occurrences of the episode {}'.format(len(occurrences), episode))
    activity = MacroActivity(episode=episode, occurrences=occurrences, events=events, period=period,
                             time_step=time_step, start_time_window_id=0, tep=tep, display=True)

    activity.fit_history_count_forecasting_model(train_ratio=0.9, method=MacroActivity.SARIMAX, display=True)

    # start_date = occurrences.date.min().to_pydatetime()
    # future_date = start_date + dt.timedelta(days=30)
    #
    # stats = activity.get_stats(8)
    #
    # print(stats)
    # pickle.dump(activity.occurrences, open("occurrences.pickle", 'wb'))


class MacroActivity:
    ID = 0

    LSTM = 0
    SARIMAX = 1

    def __init__(self, episode, occurrences, events, period, time_step, start_time_window_id, tep=30, display=False):
        '''
        Creation of a Macro-Activity
        :param episode:
        :param occurrences:
        :param events:
        :param period:
        :param time_step:
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

        # Initialize the count_histogram
        colums = ['tw_id'] + ['ts_{}'.format(ts_id) for ts_id in self.index]
        self.count_histogram = pd.DataFrame(columns=colums)  # For daily profiles

        self.duration_distrib = {}  # For activity duration laws

        for label in self.episode:
            # Stored as Gaussian Distribution
            self.duration_distrib[label] = pd.DataFrame(columns=['tw_id', 'mean', 'std'])

        self.occurrence_order = pd.DataFrame(
            columns=['tw_id'])  # For execution orders, the columns of the df are like '0132'

        self.add_time_window(occurrences, events, start_time_window_id, display=display)

        # self.build_histogram(occurrences, display=display)

        MacroActivity.ID += 1

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
            plt.title(
                '--'.join(self.episode) + '\nTime step : {} min'.format(round(self.time_step.total_seconds() / 60, 1)))
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

            for tw_id in range(max_tw_id, time_window_id):
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

    def fit_history_count_forecasting_model(self, train_ratio, method, display=False):
        """
        Fit a time series forecasting model to the history count data
        :param method:
        :return: Normalised Mean Squared Error (NMSE)
        """
        raw_dataset = self.count_histogram.drop(['tw_id'], axis=1)
        nb_tstep = len(raw_dataset.columns)

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

    def simulate(self, date, time_step_id):
        '''
        Simulate the activities on the macro-activity
        :param date:
        :param time_step_id:
        :return: simulations results, macro-activity duration
        '''

        # TODO : Fix this graph MESSSS
        graph_sim = self.graph.simulate(date, date + self.Tep)

        # graph_sim['time_before_next_event'] = graph_sim['date'].shift(-1) - graph_sim['date']
        # graph_sim['time_before_next_event'] = graph_sim['time_before_next_event'].apply(lambda x: x.total_seconds())
        # graph_sim.fillna(0, inplace=True)

        # simulation_results = pd.DataFrame(columns=['label', 'date', 'end_date'])
        #
        # for _, row in graph_sim.iterrows():
        #     activity = self.activities[row.label]
        #     _, activity_duration = activity.simulate(date, time_step_id)
        #     end_date = date + dt.timedelta(seconds=activity_duration)
        #     simulation_results.loc[len(simulation_results)] = [row.label, date, end_date]
        #     date = date + dt.timedelta(seconds=row.time_before_next_event)
        #
        # duration = (simulation_results.end_date.max().to_pydatetime() - date).total_seconds()

        duration = (graph_sim.end_date.max().to_pydatetime() - date).total_seconds()

        return graph_sim, duration




class ActivityObjectManager:
    """
    Manage the macro-activities created in the time windows
    """

    def __init__(self, name, period, time_step, tep):
        """
        Initialisation of the Manager
        :param name:
        :param period:
        :param time_step:
        """
        self.name = name
        self.period = period
        self.tep = tep
        self.time_step = time_step
        self.discovered_episodes = []  # All the episodes discovered until then
        self.activity_objects = {}  # The Activity/MacroActivity objects

    def update(self, episode, occurrences, events, time_window_id):
        """
        Update the Macro-Activity Object related to the macro-activity discovered if they already exist OR create a new Object
        :param macro_activities_list:
        :param time_window_id:
        :return:
        """

        set_episode = frozenset(episode)

        if set_episode not in self.discovered_episodes:  # Create a new Macro-Activity Object
            activity_object = MacroActivity(episode=episode, occurrences=occurrences, events=events, period=self.period,
                                            time_step=self.time_step, start_time_window_id=time_window_id, tep=self.tep,
                                            display=False)
            self.discovered_episodes.append(set_episode)
            self.activity_objects[set_episode] = activity_object
        else:
            activity_object = self.activity_objects[set_episode]
            activity_object.add_time_window(occurrences=occurrences, events=events,
                                            time_window_id=time_window_id, display=False)

    def get_MacroActivity_object(self, episode):
        """
        return the MacroActivity Object related to the episode
        :param episode:
        :return:
        """

        set_episode = frozenset(episode)

        return self.activity_objects[set_episode]

    def build_forecasting_models(self, train_ratio, method=MacroActivity.LSTM, display=False):
        """
        Build forecasting models for Macro-Activity parameters
        :param method: method to use, choose betwen LSTM, ARIMA, ...
        :return:
        """

        error_df = pd.DataFrame(columns=['episode', 'error'])
        for set_episode, macro_activity_object in self.activity_objects.items():
            print('Forecasting Model for : {}!!'.format(set_episode))
            error = macro_activity_object.fit_history_count_forecasting_model(train_ratio=train_ratio, method=method,
                                                                              display=display)
            error_df.at[len(error_df)] = [tuple(set_episode), error]

        sns.distplot(error_df.error)
        plt.show()



if __name__ == '__main__':
    main()
