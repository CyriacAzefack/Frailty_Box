import datetime as dt
import math
import random
import sys
from os.path import dirname
from subprocess import check_call

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.statespace.varmax import VARMAX

sys.path.append(dirname(__file__))

from Simulation.MacroActivity import MacroActivity


def forecast_durations(duration_data, train_ratio, nb_periods_to_forecast):
    """
    forecast the activities duration
    :param train_ratio:
    :param last_time_window_id:
    :param nb_periods_to_forecast:
    :param display:
    :return: Two dict like object {label: r2_score}
    """

    duration_dist_errors = pd.DataFrame(columns=['label', 'mean_error', 'std_error'])

    # Fill history count df until last time window registered

    last_time_window_id = int(duration_data.tw_id.max())

    forecast_df = pd.DataFrame(columns=['tw_id', 'mean', 'std'])

    forecast_df.tw_id = np.arange(last_time_window_id, last_time_window_id + nb_periods_to_forecast)

    data = duration_data[['mean', 'std']]

    train_size = int(len(data) * train_ratio)

    train, test = data[:train_size], data[train_size:]

    test_size = test.shape[0]

    model = VARMAX(train.values, order=(2, 2), enforce_stationarity=True, enforce_invertibility=False)

    try:
        model = model.fit(disp=False)
    except np.linalg.LinAlgError as e:
        return None

    raw_forecast = model.forecast(test_size + nb_periods_to_forecast)

    index = np.arange(train_size + 1, train_size + 1 + test_size + nb_periods_to_forecast)
    raw_forecast = pd.DataFrame(raw_forecast, index, ['mean', 'std'])

    validation_forecast = raw_forecast[:test_size]
    # nmse_error_mean = mean_squared_error(test['mean'], validation_forecast['mean']) / np.mean(test['mean'])
    # nmse_error_std = mean_squared_error(test['std'], validation_forecast['std']) / np.mean(test['std'])

    # Forecast to use
    forecasts = raw_forecast[test_size:]

    # Replace all negative values by 0
    forecasts[forecasts < 0] = 0

    return forecasts


class ActivityManager:
    """
    Manage the macro-activities created in the time windows
    """

    # OBSOLESCENCE_DURATION_DAYS = 60

    def __init__(self, name, period, time_step, tep, window_size=None, max_no_news=None, dynamic=False, ):
        """
        Initialisation of the Manager
        :param name:
        :param period:
        :param time_step: Time discretization parameter
        """
        self.name = name
        self.period = period
        self.tep = tep
        self.time_step = time_step
        self.window_size = window_size
        self.OBSOLESCENCE_DURATION_DAYS = max_no_news
        self.discovered_episodes = []  # All the episodes discovered until then
        self.activity_objects = {}  # The Activity/MacroActivity objects
        self.mixed_occurrences = pd.DataFrame(columns=['date', 'end_date', 'label', 'tw_id'])
        self.last_time_window_id = 0
        self.dynamic = dynamic
        self.nb_new_episodes = []
        self.labels_duration_dist = {}

    def update(self, episode, occurrences, events, time_window_id=0, display=False):
        """
        Update the Macro-Activity Object related to the macro-activity discovered if they already exist OR create a
        new Object
        :param episode:
        :param occurrences:
        :param events:
        :param time_window_id:
        :return:
        """

        set_episode = frozenset(episode)

        if set_episode not in self.discovered_episodes:  # Create a new Macro-Activity Object
            activity_object = MacroActivity(episode=episode, period=self.period, time_step=self.time_step, tep=self.tep)

            self.discovered_episodes.append(set_episode)
            self.activity_objects[set_episode] = activity_object

        activity_object = self.activity_objects[set_episode]
        activity_object.add_time_window(occurrences=occurrences, events=events,
                                        time_window_id=time_window_id, display=False)

        occurrences['label'] = [str(set_episode) for _ in range(len(occurrences))]
        occurrences['tw_id'] = time_window_id

        self.mixed_occurrences = pd.concat(
            [self.mixed_occurrences, occurrences[['date', 'end_date', 'label', 'tw_id']]]).drop_duplicates(keep=False)

        self.mixed_occurrences.sort_values(['date'], inplace=True)

        self.last_time_window_id = time_window_id

        if self.dynamic:
            self.obsolescence_check()

        self.nb_new_episodes.append(len(self.discovered_episodes))

    def get_macro_activity_from_name(self, set_episode):
        """
        return the MacroActivity Object related to the episode
        :param set_episode:
        :return:
        """

        # set_episode = frozenset(set_episode)

        return self.activity_objects[set_episode]

    def build_transition_matrix(self, time_window_id=0, display=False):
        """
        Build a transition distance_matrix for all the available activities
        :return:
        """

        data = self.mixed_occurrences[self.mixed_occurrences.tw_id == time_window_id].copy()
        labels = data.label.unique()

        data['next_label'] = data['label'].shift(-1)
        data.dropna(inplace=True)

        matrix = pd.DataFrame(columns=labels, index=labels)

        for i_label in labels:
            i_data = data[data.label == i_label]
            nb_i_occ = len(i_data)
            for j_label in labels:
                nb_j_occ = len(i_data[i_data.next_label == j_label])
                matrix.loc[i_label, j_label] = nb_j_occ / nb_i_occ

        if display:
            plot_markov_chain(matrix, labels, threshold=0.0)

        return matrix

    def obsolescence_check(self):
        """
        Remove Macro-Activities which are obsoletes
        :return:
        """

        obsoletes_episodes = []

        for set_episode, macro_activity_object in self.activity_objects.items():
            # Computer Macro-Activity Weight !!
            last_update_id = int(macro_activity_object.count_histogram.tw_id.max())

            nb_period_without_news = self.last_time_window_id - last_update_id

            if nb_period_without_news > self.OBSOLESCENCE_DURATION_DAYS:
                obsoletes_episodes.append(set_episode)
                print(print(f"{list(set_episode)} : OBSOLETE EPISODE"))

        # DESTROY OBSOLETE EPISODES
        for set_episode in obsoletes_episodes:
            self.activity_objects.pop(set_episode)
            self.discovered_episodes.remove(set_episode)

    def build_forecasting_models(self, train_ratio, nb_periods_to_forecast, display=False, debug=False):
        """
        Build forecasting models for Macro-Activity parameters
        :param debug:
        :param nb_periods_to_forecast:
        :param train_ratio: ratio of data used for training
        :param method: method used for the forecasting
        :param display:
        :return:
        """

        ADP_error_df = pd.DataFrame(columns=['episode', 'rmse'])
        duration_error_df = pd.DataFrame(columns=['episode', 'mean_rmse', 'std_rmse', 'label'])

        # Build forecasting models
        i = 0

        for set_episode, macro_activity_object in self.activity_objects.items():
            print(f"{list(set_episode)} : {macro_activity_object}")

            i += 1

            # ACTIVITY DAILY PROFILE FORECASTING
            # TODO : Enable ADP Forecasting

            # Fill

            ADP_error = macro_activity_object.forecast_history_count(train_ratio=train_ratio,
                                                                     last_time_window_id=self.last_time_window_id,
                                                                     nb_periods_to_forecast=nb_periods_to_forecast,
                                                                     display=debug)
            #
            # ADP_error_df.at[len(ADP_error_df)] = [tuple(set_episode), ADP_error]

            # ACTIVITIES DURATIONS FORECASTING
            # TODO : Monitor the error on forecasting models for duration
            for label in macro_activity_object.episode:
                macro_activity_object.duration_distrib[label] = self.labels_duration_dist[label]
            #
            # error_df = macro_activity_object.forecast_durations(train_ratio=train_ratio,
            #                                                     last_time_window_id=self.last_time_window_id,
            #                                                     nb_periods_to_forecast=nb_periods_to_forecast,
            #                                                     display=debug)
            #
            # for _, row in error_df.iterrows():
            #     duration_error_df.loc[len(duration_error_df)] = [tuple(set_episode), row['mean_error'],
            #                                                      row['std_error'], row['label']]

            # STOP HERE IF SINGLE-ACTIVITY
            # if len(set_episode) < 2:
            #     break

            # EXECUTION ORDER FORECASTING
            # exec_order_error = macro_activity_object.forecast_execution_order(
            #     train_ratio=train_ratio, last_time_window_id=self.last_time_window_id,
            #     nb_periods_to_forecast=nb_periods_to_forecast, debug=debug)

            sys.stdout.write(
                "\r{}/{} Macro-Activities Forecasting models done...".format(i, len(self.activity_objects)))
            sys.stdout.flush()
        sys.stdout.write("\n")

        if display:
            # Drop NAN
            ADPs_rmse = ADP_error_df.replace([np.inf, -np.inf], np.nan).dropna().rmse.values
            mean_durations_rmse = duration_error_df.replace([np.inf, -np.inf], np.nan).dropna().mean_rmse.values
            std_durations_rmse = duration_error_df.replace([np.inf, -np.inf], np.nan).dropna().std_rmse.values

            sns.kdeplot(ADPs_rmse, shade_lowest=False, shade=True, label='ADP Errors')
            plt.show()
            sns.kdeplot(mean_durations_rmse, shade_lowest=False, shade=True, label='Mean Duration Errors')
            sns.kdeplot(std_durations_rmse, shade_lowest=False, shade=True, label='STD Duration Errors')
            plt.show()

        # plt.hist(list(error_df.error.values))
        # plt.title('NMSE Distribution for all macro_activities forecasting models')
        # plt.show()

        return ADP_error_df, duration_error_df

    def build_forecasting_duration(self, dataset, train_ratio, nb_periods_to_forecast):
        """
        :param train_ratio:
        :param nb_periods_to_forecast:
        :param display:
        :param debug:
        :return:
        """
        time_window_duration = dt.timedelta(days=self.window_size)
        start_date = dataset.date.min().to_pydatetime()
        end_date = dataset.date.max().to_pydatetime() - time_window_duration

        dataset['duration'] = dataset.end_date - dataset.date
        dataset['duration'] = dataset['duration'].apply(lambda x: x.total_seconds())

        labels = dataset.label.unique()

        labels_duration_dist = {}
        for label in labels:
            df = pd.DataFrame(columns=['tw_id', 'mean', 'std'])
            labels_duration_dist[label] = df

        nb_tw = math.floor((end_date - start_date) / self.period)

        for tw_id in range(nb_tw):
            tw_start_date = start_date + dt.timedelta(days=tw_id)
            tw_end_date = tw_start_date + dt.timedelta(days=self.window_size)

            tw_dataset = dataset[(dataset.date >= tw_start_date) & (dataset.date < tw_end_date)]

            for label in labels:
                mean_duration = np.mean(tw_dataset[tw_dataset.label == label]['duration'])
                std_duration = np.std(tw_dataset[tw_dataset.label == label]['duration'])

                df = labels_duration_dist[label]
                df.loc[tw_id] = [tw_id, mean_duration, std_duration]

        for label in labels:
            label_data = labels_duration_dist[label]

            forecast_data = forecast_durations(label_data, train_ratio, nb_periods_to_forecast)

            if forecast_data is None:
                forecast_data = pd.DataFrame(columns=['tw_id', 'mean', 'std'])
                forecast_data['tw_id'] = np.arange(len(label_data), len(label_data) + nb_periods_to_forecast)
                forecast_data['mean'] = [label_data['mean'].values[-1]] * nb_periods_to_forecast
                forecast_data['std'] = [label_data['std'].values[-1]] * nb_periods_to_forecast
                label_data = label_data.append(forecast_data, ignore_index=True)
            else:
                forecast_data['tw_id'] = np.arange(len(label_data), len(label_data) + nb_periods_to_forecast)
                label_data = label_data.append(forecast_data, ignore_index=True)

            label_data.fillna(0, inplace=True)

        self.labels_duration_dist = labels_duration_dist

    def get_activity_daily_profiles(self, time_window_id=0):
        """
        :param time_window_id:
        :return: a dict-like object of Macro-activities ADP
        """

        macro_ADPs = {}

        activities_count_histogram = pd.DataFrame()

        for set_episode, macro_activity in self.activity_objects.items():
            count_histogram = macro_activity.get_count_histogram(time_window_id=time_window_id)
            count_histogram.drop(['tw_id'], axis=1, inplace=True)
            count_histogram.index = [set_episode]
            activities_count_histogram = activities_count_histogram.append(count_histogram)

        # No 'idle' times
        # activities_count_histogram = activities_count_histogram.div(activities_count_histogram.sum(axis=0), axis=1)

        # With 'idle' times

        div_array = activities_count_histogram.sum(axis=0)

        div_array = np.where(div_array > self.window_size, div_array, self.window_size)

        activities_occ_probability = activities_count_histogram.div(div_array, axis=1)

        activities_occ_probability.fillna(0, inplace=True)

        for index, row in activities_occ_probability.iterrows():
            macro_ADPs[index] = row.values

        return macro_ADPs

    def simulate(self, start_date, end_date, idle_duration, time_window_id=0):
        """
        Generate data between two dates using the model parameters for the selected time_window_id
        :param start_date: Start date of the simulation
        :param end_date: end date
        :param idle_duration: duration of a idle period of time
        :param time_window_id: selected time window id
        :return:
        """

        simulated_dataset = pd.DataFrame(columns=['date', 'end_date', 'label'])

        # Use
        previous_event = None
        current_date = start_date

        simulation_duration = (end_date - start_date).total_seconds()

        macro_ADPs = self.get_activity_daily_profiles(time_window_id=time_window_id)

        # transition_matrix = self.build_transition_matrix(time_window_id=time_window_id)

        current_time_window_id = time_window_id

        while current_date < end_date:

            evolution_percentage = round(100 * ((current_date - start_date).total_seconds() / simulation_duration), 2)
            sys.stdout.write("\r{} %% of Simulation done!!".format(evolution_percentage))
            sys.stdout.flush()

            # Compute the time window id
            if self.dynamic:
                new_time_window_id = time_window_id + int((current_date - start_date) / self.period)
                if current_time_window_id > new_time_window_id:  # if we change time window
                    # Re compute the Activity Daily Profiles
                    current_time_window_id = new_time_window_id
                    macro_ADPs = self.get_activity_daily_profiles(time_window_id=self.last_time_window_id)
            else:
                current_time_window_id = time_window_id

            # Compute the time step id

            day_date = dt.datetime.combine(pd.to_datetime(current_date).date(), dt.datetime.min.time())

            time_step_id = math.ceil((current_date - day_date).total_seconds() \
                                     / self.time_step.total_seconds()) % int(self.period / self.time_step)

            # Choose the next Activity

            set_episodes = []
            scores_episodes = []

            for set_episode, macro_activity in self.activity_objects.items():
                # # Transition probability
                # if previous_event == None:
                #     prob_score = 1
                # else:
                #     prob_score = transition_matrix.loc[str(previous_event.get_set_episode())][str(set_episode)]

                # ADP score
                ADP_value = macro_ADPs[set_episode][time_step_id]
                # ADP_score = ADP_value if rand > ADP_value else 1

                # TODO : Linear function for Activity selection
                score = ADP_value

                set_episodes.append(set_episode)
                scores_episodes.append(score)

            scores_episodes = np.asarray(scores_episodes)

            scores_episodes = scores_episodes / sum(scores_episodes)

            scores_episodes = np.cumsum(scores_episodes)

            rand = random.random()

            chosen_set_episode = None
            for i in range(len(scores_episodes)):
                if rand <= scores_episodes[i]:
                    chosen_set_episode = set_episodes[i]
                    break

            # # Pick the episode with the max score
            # chosen_set_episode = max(scores.items(), key=operator.itemgetter(1))[0]

            if chosen_set_episode is None:  # Nothing happens
                current_date += idle_duration
                continue

            chosen_macro_activity = self.get_macro_activity_from_name(chosen_set_episode)

            # print("Time step ID : {}".format(time_step_id))
            # print("Chosen Activity : {} ".format(chosen_macro_activity))

            # Simulate the MACRO-ACTIVITY
            macro_activity_events = chosen_macro_activity.simulate(start_date=current_date, time_step_id=time_step_id,
                                                                   time_window_id=current_time_window_id)
            simulated_dataset = simulated_dataset.append(macro_activity_events)

            current_date = simulated_dataset.end_date.max().to_pydatetime()

            # previous_event = chosen_macro_activity

        sys.stdout.write("\n")

        return simulated_dataset

    def dump_data(self, output):
        """
        Dump all the data related to the macro-activities
        :param output:
        :return:
        """
        for set_episode, macro_activity in self.activity_objects.items():
            macro_activity.dump_data(output=output)

    def ADP_screenshot(self, time_window_id=0):
        """
        Screenshot of all the macro-activities daily profiles
        :param time_window_id:
        :return:
        """

        macro_ADPs = self.get_activity_daily_profiles(time_window_id=time_window_id)

        df_ADPs = pd.DataFrame.from_dict(macro_ADPs, orient='index')

        print()


def plot_markov_chain(matrix, labels, threshold=0.1):
    """
    Plot the markov chain corresponding to the distance_matrix
    :param matrix:
    :param labels:
    :return:
    """
    G = nx.MultiDiGraph()
    G.add_nodes_from(labels)
    print(f'Nodes:\n{G.nodes()}\n')

    edges_wts = {}
    for col in matrix.columns:
        for idx in matrix.index:
            if matrix.loc[idx, col] > threshold:
                edges_wts[(idx, col)] = round(matrix.loc[idx, col], 2)

    # edges represent transition probabilities
    for k, v in edges_wts.items():
        tmp_origin, tmp_destination = k[0], k[1]
        G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
    nx.draw_networkx(G, pos)

    edge_labels = {(n1, n2): d['label'] for n1, n2, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    filename = 'markov'
    nx.drawing.nx_pydot.write_dot(G, filename + '.dot')
    check_call(['dot', '-Tpng', filename + '.dot', '-o', filename + '.png'])

    plt.clf()
    plt.axis('off')
    img = mpimg.imread(filename + '.png')
    plt.imshow(img)
    plt.title('Markov Chain between activities')
    plt.show()
