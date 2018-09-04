import seaborn as sns
from fbprophet import Prophet
from pylab import plt

from Graph_Model import Acyclic_Graph
from Graph_Model.Pattern2Graph import *
from xED.Candidate_Study import modulo_datetime, find_occurrences
from xED.Pattern_Discovery import pick_dataset

sns.set_style('darkgrid')


def main():
    dataset = pick_dataset('aruba')

    # path = "C:/Users/cyriac.azefack/Workspace/Frailty_Box/output/Simulation results 1/dataset_simulation_10.csv"
    # dataset = pick_custom_dataset(path)

    label = ('sleeping',)
    period = dt.timedelta(days=1)
    time_step = dt.timedelta(minutes=15)
    occurrences = find_occurrences(data=dataset, episode=label)

    print('{} occurrences of the episode {}'.format(len(occurrences), label))
    activity = Activity(label=label, occurrences=occurrences, period=period, time_step=time_step,
                        display_histogram=True)

    # start_date = occurrences.date.min().to_pydatetime()
    # future_date = start_date + dt.timedelta(days=30)
    #
    # stats = activity.get_stats(8)
    #
    # print(stats)
    # pickle.dump(activity.occurrences, open("occurrences.pickle", 'wb'))


class Activity:
    ID = 0
    SLIDING_WINDOW = dt.timedelta(days=30)

    def __init__(self, label, occurrences, period, time_step, start_date, end_date, duration_gen='Normal',
                 display_histogram=False):
        '''
        Creation of an activity
        :param label: label of the activity
        :param occurrences: Dataframe representing the occurrences of the activity
        :param period: [dt.timedelta] Frequency of the analysis
        :param time_step: [dt.timedelta] discret time step for the simulation
        '''

        print('\n')
        print("####################################")
        print(" Creation of the Activity '{}'".format(label))
        print("####################################")
        print('\n')
        self.label = label
        self.period = period
        self.time_step = time_step
        self.duration_gen = duration_gen
        self.index = np.arange(int(period.total_seconds() / time_step.total_seconds()) + 1)
        self.occurrences = self.preprocessing(occurrences)
        self.histogram = self.build_histogram(occurrences, display=display_histogram)
        self.activity_duration_model = self.build_activity_duration_model(occurrences)
        # self.time_evo_per_index = self.compute_time_evolution()

        if self.duration_gen == 'TS Forecast':
            self.duration_forecasts = self.build_duration_forecaster(start_date, end_date)
            # self.forecasters_per_index = self.build_time_step_forecasters()

        # evo = self.time_evo_per_index[3]
        # plt.plot(evo.index, evo.mean_duration)
        # plt.show()

        Activity.ID += 1

    def preprocessing(self, occurrences):
        '''
        Preprocessing the occurrences
        :param occurrences:
        :return:
        '''
        occurrences['relative_date'] = occurrences.date.apply(
            lambda x: modulo_datetime(x.to_pydatetime(), self.period))
        occurrences['time_step_id'] = occurrences['relative_date'] / self.time_step.total_seconds()
        occurrences['time_step_id'] = occurrences['time_step_id'].apply(math.floor)
        occurrences['activity_duration'] = occurrences.end_date - occurrences.date
        occurrences['activity_duration'] = occurrences['activity_duration'].apply(lambda x: x.total_seconds())

        return occurrences

    def build_histogram(self, occurrences, display=False):
        '''
        Build the Time distribution histogram on occurrences
        :param occurrences:
        :param display:
        :return:
        '''

        hist = occurrences.groupby(['time_step_id']).count()['date']

        # Create an index to have every time steps in the period

        hist = hist.reindex(self.index)
        hist.fillna(0, inplace=True)

        if display:
            hist.plot(kind="bar")
            plt.title(
                '--'.join(self.label) + '\nTime step : {} min'.format(round(self.time_step.total_seconds() / 60, 1)))
            plt.ylabel('Count')
            plt.show()

        return hist

    def build_activity_duration_model(self, occurrences):
        '''
        Build Activity duration Model
        :param occurrences:
        :return:
        '''

        aggregation = {
            'activity_duration': {
                'mean_duration': 'mean',
                'std_duration': 'std'
            }
        }
        activity_duration = occurrences[['time_step_id', 'activity_duration']].groupby(['time_step_id']).agg(
            aggregation)
        activity_duration = activity_duration.reindex(self.index)
        activity_duration.fillna(0, inplace=True)
        activity_duration.columns = activity_duration.columns.droplevel(level=0)

        return activity_duration

    def build_time_step_forecasters(self):
        '''
        Build the time series forecasters_per_index
        :return: a dict like {'time_step_id' : {'hist_count': ..., 'mean_duration': ..., 'std_duration':...}, ...}
        '''

        forecasters_per_index = [{
            'hist_count': None,  # Prophet forecasters_per_index for each indicator
            'mean_duration': None,
            'std_duration': None
        } for i in self.index]

        # forecasting_pre_computing

        for time_step_id in self.index:
            time_step_forecaster = forecasters_per_index[time_step_id]
            df = self.time_evo_per_index[time_step_id]
            df['date'] = df.index

            for indicator_name in time_step_forecaster.keys():
                df_indicator = df[['date', indicator_name]].copy()
                df_indicator.columns = ['ds', 'y']

                with suppress_stdout_stderr():
                    if df_indicator.empty:
                        time_step_forecaster[indicator_name] = None
                        continue
                    time_step_forecaster[indicator_name] = Prophet().fit(df_indicator)

            evolution = (time_step_id + 1) / len(self.index)
            evolution_percentage = round(100 * evolution, 2)
            sys.stdout.write("\r{} %% of Forecasters built !!".format(evolution_percentage))
            sys.stdout.flush()
        print()
        return forecasters_per_index

    def build_duration_forecaster(self, start_date, end_date):
        '''
        Build a Time Series Forecaster for activity duration
        :return:
        '''
        df = self.occurrences[['date', 'activity_duration']]
        df.columns = ['ds', 'y']

        with suppress_stdout_stderr():
            forecaster = Prophet().fit(df)

        # TODO : Replace this by the actual future

        start_date = start_date - self.period - dt.timedelta(seconds=modulo_datetime(start_date, self.period))

        end_date = end_date + self.period - dt.timedelta(seconds=modulo_datetime(end_date, self.period))

        future = pd.date_range(start_date, end_date, freq='{}S'.format(self.time_step.total_seconds()))
        future = pd.DataFrame({'ds': future})

        forecast = forecaster.predict(future)

        forecast = forecast[['ds', 'yhat']]
        forecast.columns = ['date', 'pred_duration']

        forecast['relative_date'] = forecast.date.apply(
            lambda x: modulo_datetime(x.to_pydatetime(), self.period))
        forecast['time_step_id'] = forecast['relative_date'] / self.time_step.total_seconds()
        forecast['time_step_id'] = forecast['time_step_id'].apply(math.floor)

        forecast['day_date'] = forecast.date.dt.date

        return forecast

    def get_stats_from_date(self, date, time_step_id, hist=True):
        '''
        Estimate the number of time the current activity started at time_step_id in the last Sliding_Window
        :param date:
        :param time_step_id:
        :param hist: if True, return only the Hist_count, otherwhise everything
        :return: a dict with the value of indicators {'hist_count':..., 'mean_duration':..., 'std_duration':...}
        '''

        if hist:
            indicators = ['hist_count']
        else:
            indicators = ['mean_duration', 'std_duration']

        stats = {
            'hist_count': None,
            'mean_duration': None,
            'std_duration': None
        }

        date = date.date()
        time_step_forecaster = self.forecasters_per_index[time_step_id]

        # future_start_date = self.time_evo_per_index[time_step_id].index.max().to_pydatetime()
        future_end_date = date
        # future = pd.date_range(future_start_date, future_end_date, freq='1D')
        future = pd.DataFrame(np.array([future_end_date], dtype=np.datetime64))

        future.columns = ['ds']

        for indicator in indicators:
            forecaster = time_step_forecaster[indicator]
            forecast = forecaster.predict(future)
            row_prediction = forecast.iloc[-1]
            indicator_value = np.random.triangular(row_prediction.yhat_lower, row_prediction.yhat,
                                                   row_prediction.yhat_upper)
            indicator_value = max(0, indicator_value)
            stats[indicator] = indicator_value

        return stats

    def get_stats(self, time_step_id):
        '''
        Get parameters at this time_step_id
        :param time_step_id:
        :return:
        '''
        stats = {
            'hist_count': None,
            'mean_duration': None,
            'std_duration': None
        }

        stats['hist_count'] = self.histogram.loc[time_step_id]
        stats['mean_duration'] = self.activity_duration_model.loc[time_step_id].mean_duration
        stats['std_duration'] = self.activity_duration_model.loc[time_step_id].std_duration

        return stats

    def compute_time_evolution(self):
        '''
        Compute the time evolution of each time_step_id
        :return:
        '''

        start_date = self.occurrences.date.min().to_pydatetime()
        # We start at the beginning of the first period
        start_date = start_date - dt.timedelta(seconds=modulo_datetime(start_date, self.period))
        end_date = self.occurrences.date.max().to_pydatetime()
        end_date = end_date + self.period - dt.timedelta(seconds=modulo_datetime(end_date, self.period))
        end_date = end_date - Activity.SLIDING_WINDOW

        nb_days_per_period = self.period.days
        time_evo_per_index = [
            pd.DataFrame(index=pd.date_range(start_date, end_date, freq=str(nb_days_per_period) + 'D'),
                         columns=['hist_count', 'mean_duration', 'std_duration']).fillna(0) for i in self.index]

        current_date = start_date

        while current_date <= end_date:
            window_start_date = current_date
            window_end_date = window_start_date + Activity.SLIDING_WINDOW

            window_occurrences = self.occurrences.loc[(
                                                              self.occurrences.date >= window_start_date) & (
                                                              self.occurrences.date < window_end_date)].copy()

            hist = self.build_histogram(window_occurrences)
            activity_durations = self.build_activity_duration_model(window_occurrences)

            for i in self.index:
                time_step_df = time_evo_per_index[i]
                hist_count = hist.loc[i]
                mean_duration = activity_durations.loc[i].mean_duration
                std_duration = activity_durations.loc[i].std_duration

                time_step_df.loc[current_date] = [hist_count, mean_duration, std_duration]

            evolution = (current_date - start_date).total_seconds() / (end_date - start_date).total_seconds()

            evolution_percentage = round(100 * evolution, 2)
            sys.stdout.write("\r{} %% of Time evolution computed !!".format(evolution_percentage))
            sys.stdout.flush()

            current_date += self.period
        sys.stdout.write("\n")

        return time_evo_per_index

    def simulate(self, date, time_step_id):
        """
        Generate the events for the activity
        :param date:
        :param time_step_id:
        :return:
        """

        simulation_result = pd.DataFrame(columns=['date', 'end_date', 'label'])

        while True:  # To prevent cases where the date is OutOfDatetimeBounds

            generated_duration = self.duration_generation(date, time_step_id, method=self.duration_gen)

            if generated_duration == 0:
                break

            # print("Time spent for prediction: {}".format(
            #     dt.timedelta(seconds=round(t.process_time() - start_time, 1))))
            try:
                event_start_date = date
                event_end_date = event_start_date + dt.timedelta(seconds=generated_duration)
                simulation_result.loc[len(simulation_result)] = [event_start_date, event_end_date,
                                                                 '--'.join(self.label)]
                break
            except ValueError as er:
                print("OOOps ! Date Overflow. Let's try again...")

        return simulation_result, generated_duration


    def duration_generation(self, date, ts_id, method='Normal'):
        """
        Generate the duration of the activity
        :param date:
        :param ts_id:
        :param method:
            'Normal' : Static normal distribution of the duration at each time step id
            'Forecast Normal' : Static normal distribution of the duration at each time step id forecasted for this specific date
            'TS Forecast' : Use a Time series forecasting model to predict the duration at this specific date (by one time step)
        :return:
        """
        generated_duration = -1

        if method == 'Normal':
            stats = self.get_stats(time_step_id=ts_id)
            mean_duration = stats['mean_duration']
            std_duration = stats['std_duration']
            generated_duration = np.random.normal(mean_duration, std_duration)

        elif method == 'Forecast Normal':
            stats = self.get_stats_from_date(date=date, time_step_id=ts_id, hist=False)
            mean_duration = stats['mean_duration']
            std_duration = stats['std_duration']
            generated_duration = np.random.normal(mean_duration, std_duration)

        elif method == 'TS Forecast':
            day_date = date.date()
            relative_date = modulo_datetime(date, self.period)
            time_step_id = math.floor(relative_date / self.time_step.total_seconds())
            row = self.duration_forecasts.loc[(self.duration_forecasts.day_date == day_date) & (
                    self.duration_forecasts.time_step_id == time_step_id), 'pred_duration']
            generated_duration = row.values[0]

        if generated_duration < 0:
            generated_duration = 0

        return generated_duration


class MacroActivity(Activity):

    def __init__(self, episode, dataset, occurrences, period, time_step, start_date, end_date, duration_gen='Normal',
                 display=False, Tep=30):
        '''
        Create a Macro Activity
        :param episode:
        :param dataset:
        :param occurrences:
        :param period:
        :param time_step:
        :param start_date:
        :param end_date:
        :param display:
        :param Tep:
        '''
        Activity.__init__(self, episode, occurrences, period, time_step, start_date, end_date,
                          duration_gen, display_histogram=display)
        self.Tep = dt.timedelta(minutes=30)
        # Find the events corresponding to the occurrences
        events = pd.DataFrame(columns=["date", "label", 'occ_id'])
        for index, occurrence in occurrences.iterrows():
            occ_start_date = occurrence["date"]
            occ_end_date = occ_start_date + dt.timedelta(minutes=Tep)
            mini_data = dataset.loc[(dataset.label.isin(episode))
                                    & (dataset.date >= occ_start_date)
                                    & (dataset.date < occ_end_date)].copy()
            mini_data.sort_values(["date"], ascending=True, inplace=True)
            mini_data.drop_duplicates(["label"], keep='first', inplace=True)
            mini_data['occ_id'] = index
            events = events.append(mini_data, ignore_index=True)

        self.activities = {}  # key: label, value: Activity

        for label in episode:
            label_events = events.loc[events.label == label].copy()
            label_activity = Activity(label=(label,), occurrences=label_events, period=period,
                                      duration_gen=duration_gen, time_step=time_step, start_date=start_date,
                                      end_date=end_date)
            self.activities[label] = label_activity

        # TODO : Build a graph for every time_step_id
        self.graph = self.build_activities_graph(episode=episode, events=events, period=period, display=display)

    def simulate(self, date, time_step_id):
        '''
        Simulate the activities on the macro-activity
        :param date:
        :param time_step_id:
        :return: simulations results, macro-activity duration
        '''

        # TODO : Fix this graph MESSSS
        graph_sim = self.graph.simulate(date, date + self.Tep)

        graph_sim['time_before_next_event'] = graph_sim['date'].shift(-1) - graph_sim['date']
        graph_sim['time_before_next_event'] = graph_sim['time_before_next_event'].apply(lambda x: x.total_seconds())
        graph_sim.fillna(0, inplace=True)

        simulation_results = pd.DataFrame(columns=['label', 'date', 'end_date'])

        for _, row in graph_sim.iterrows():
            activity = self.activities[row.label]
            _, activity_duration = activity.simulate(date, time_step_id)
            end_date = date + dt.timedelta(seconds=activity_duration)
            simulation_results.loc[len(simulation_results)] = [row.label, date, end_date]
            date = date + dt.timedelta(seconds=row.time_before_next_event)

        duration = (simulation_results.end_date.max().to_pydatetime() - date).total_seconds()

        return simulation_results, duration

    def build_activities_graph(self, episode, events, period, display=False):
        '''
        Create a graph for the Macro_Activities
        :param episode:
        :param events:
        :param p:
        '''

        occurrence_ids = events['occ_id'].unique()

        # Build a list of occurrences events list to build the graph
        events_occurrences_lists = []

        graph_nodes_labels = []
        for occ_id in occurrence_ids:
            occ_list = []
            occ_df = events[events.occ_id == occ_id]
            i = 0
            for index, event_row in occ_df.iterrows():
                new_label = event_row['label'] + '_' + str(i)
                events.at[index, 'label'] = new_label
                occ_list.append(new_label)
                i += 1
            graph_nodes_labels += occ_list
            events_occurrences_lists.append(occ_list)

        # Set of graph_nodes for the graphs
        graph_nodes_labels = set(graph_nodes_labels)

        for occ_id in range(min(occurrence_ids), max(occurrence_ids) + 1):
            events_occurrences_lists.append(events.loc[events.occ_id == occ_id, 'label'].tolist())

        graph_nodes, graph_labels, prob_matrix = build_probability_acyclic_graph(list(episode), graph_nodes_labels,
                                                                                 events_occurrences_lists)

        # Build the time matrix
        events.loc[:, "relative_date"] = events.date.apply(
            lambda x: modulo_datetime(x.to_pydatetime(), period))
        events['is_last_event'] = events['occ_id'] != events['occ_id'].shift(-1)
        events['is_first_event'] = events['occ_id'] != events['occ_id'].shift(1)
        events['next_label'] = events['label'].shift(-1).fillna('_nan').apply(Acyclic_Graph.Acyclic_Graph.node2label)
        events['next_date'] = events['date'].shift(-1)
        events['inter_event_duration'] = events['next_date'] - events['date']
        events['inter_event_duration'] = events['inter_event_duration'].apply(lambda x: x.total_seconds())
        # events = events[events.is_last_event == False]

        n = len(graph_nodes)  # Nb rows of the prob matrix
        l = len(graph_labels)  # Nb columns of the prob matrix

        # n x l edges for waiting time transition laws
        time_matrix = [[[] for j in range(l)] for i in
                       range(n)]  # Empty lists, [[mean_time, std_time], ...] transition durations

        for i in range(n):
            for j in range(l - 1):  # We dont need the "END NODE"
                if prob_matrix[i][j] != 0:  # Useless to compute time for never happening transition
                    from_node = graph_nodes[i]
                    to_label = graph_labels[j]

                    if from_node == Acyclic_Graph.Acyclic_Graph.START_NODE:  # START_NODE transitions
                        time_matrix[i][j] = ('norm', [0, 0])
                        continue

                    time_df = events.loc[(events.label == from_node) & (events.next_label == to_label)]
                    inter_events_durations = time_df.inter_event_duration.values

                    # We remove NaN from the values
                    inter_events_durations = inter_events_durations[~np.isnan(inter_events_durations)]
                    inter_events_durations = clean_data_arrays(inter_events_durations)
                    time_matrix[i][j] = ('norm', [np.mean(inter_events_durations), np.std(inter_events_durations)])

        events['activity_duration'] = events['end_date'] - events['date']
        events['activity_duration'] = events['activity_duration'].apply(lambda x: x.total_seconds())

        duration_matrix = [[] for i in range(n)]  # Empty lists, [[mean_time, std_time], ...] Activity duration
        for i in range(n):
            node = graph_nodes[i]
            if node != Acyclic_Graph.Acyclic_Graph.START_NODE:
                time_df = events.loc[events.label == node]
                activity_durations = time_df.activity_duration.values
                # We remove NaN from the values
                activity_durations = activity_durations[~np.isnan(activity_durations)]
                if len(activity_durations) > 0:
                    activity_durations = clean_data_arrays(activity_durations)
                    # plt.figure()
                    # sns.distplot(activity_durations)
                    # plt.show()
                    duration_matrix[i] = ('norm', [np.mean(activity_durations), np.std(activity_durations)])

        acyclic_graph = Acyclic_Graph.Acyclic_Graph(nodes=graph_nodes, labels=list(episode), period=period,
                                                    prob_matrix=prob_matrix,
                                                    wait_matrix=time_matrix, activities_duration=duration_matrix)

        if display:
            acyclic_graph.display(output_folder='./', debug=True)

        return acyclic_graph


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
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


if __name__ == '__main__':
    main()
