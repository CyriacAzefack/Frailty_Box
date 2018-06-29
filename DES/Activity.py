import datetime as dt
import math
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from fbprophet import Prophet
from pylab import exp, sqrt, diag, plot, legend, plt
from scipy.optimize import curve_fit

from xED.Candidate_Study import modulo_datetime, find_occurrences
from xED.Pattern_Discovery import pick_dataset

sns.set_style('darkgrid')


def main():
    dataset = pick_dataset('aruba')

    # path = "C:/Users/cyriac.azefack/Workspace/Frailty_Box/output/Simulation results 1/dataset_simulation_10.csv"
    # dataset = pick_custom_dataset(path)

    label = ('meal_preparation',)
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

    def __init__(self, label, occurrences, period, time_step, forecast_precomputing=False, display_histogram=False):
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
        self.index = np.arange(int(period.total_seconds() / time_step.total_seconds()) + 1)
        self.occurrences = self.preprocessing(occurrences)
        self.histogram = self.build_histogram(occurrences, display=display_histogram)
        self.activity_duration_model = self.build_activity_duration_model(occurrences)
        # self.time_evo_per_index = self.compute_time_evolution()

        if forecast_precomputing:
            self.forecasters_per_index = self.build_forecaster()

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
        Build the histogram on occurrences
        :param occurrences:
        :param period:
        :param time_step:
        :return:
        '''

        hist = occurrences.groupby(['time_step_id']).count()['date']

        # Create an index to have every time steps in the period

        hist = hist.reindex(self.index)
        hist.fillna(0, inplace=True)

        if display:
            hist.plot(kind="bar")
            plt.title('--'.join(self.label))

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

    def build_forecaster(self):
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
        :param hist:
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
        :param data:
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
        '''
        Generate the events for the activity
        :param date:
        :param time_step_id:
        :return:
        '''

        # stats = chosen_activity.get_stats_from_date(date=current_date, time_step_id=time_step_id, hist=False)
        stats = self.get_stats(time_step_id=time_step_id)
        mean_duration = stats['mean_duration']
        std_duration = stats['std_duration']

        simulation_result = pd.DataFrame(columns=['date', 'end_date', 'label'])
        while True:
            generated_duration = -1
            while generated_duration < 0:
                generated_duration = np.random.normal(mean_duration, std_duration)

            try:
                event_start_date = date
                event_end_date = event_start_date + dt.timedelta(seconds=generated_duration)
                simulation_result.loc[len(simulation_result)] = [event_start_date, event_end_date,
                                                                 '--'.join(self.label)]
                break
            except ValueError as er:
                print("OOOps ! Date Overflow. Let's try again...")

        return simulation_result, generated_duration

    def fit_bimodal_distribution(self, x, y, index):

        def gauss(x, mu, sigma, A):
            return A * exp(-(x - mu) ** 2 / 2 / sigma ** 2)

        def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3):
            return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2) + gauss(x, mu3, sigma3, A3)

        # expected = (1, .2, 250, 2, .2, 125)
        m = np.min(index)
        M = np.max(index)
        mid = int((m + M) / 2)
        expected = (m, 1, 1, mid, 1, 1, M, 1, 1)
        bound_min = [-M, 0, 0, -M, 0, 0, -M, 0, 0]
        bound_max = [2 * M, mid, np.inf, 2 * M, mid, np.inf, 2 * M, mid, np.inf]

        params, cov = curve_fit(bimodal, x, y, None, bounds=(bound_min, bound_max))
        sigma = sqrt(diag(cov))
        plot(x, bimodal(x, *params), color='red', lw=3, label='model')
        legend()
        print(params, '\n', sigma)
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
