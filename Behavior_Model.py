import datetime as dt
import errno
import math
import multiprocessing as mp
import os
import pickle
import sys
import time as t
from optparse import OptionParser

import seaborn as sns

import Pattern_Mining
import Utils
# from Pattern_Mining.Candidate_Study import find_occurrences, modulo_datetime
from Pattern_Mining.Extract_Macro_Activities import extract_macro_activities
from Simulation import ActivityManager

sns.set(font_scale=1.8)


# random.seed(1996)


def main(args):
    ####################
    # Macro Activities MODEL
    #
    # Characteristics :
    #   - Single Activity Implementation
    #   - No time evolution
    #   - Training Dataset : 80% of the Original log_dataset
    #   - Test Dataset : 20% left of the Original    log_dataset

    period = dt.timedelta(days=1)

    parser = OptionParser(usage='Usage of the Digital Twin Simulation model algorihtm: %prog <options>')
    parser.add_option('-n', '--dataset_name', help='Name of the Input event log', dest='dataset_name', action='store',
                      type='string')
    parser.add_option('--static_learning', help='Time evolution not taken into account', dest='static_learning',
                      action='store_true',
                      default=False)
    parser.add_option('--window_days', help='Number of days per time windows', dest='window_days', type=int, default=30)
    parser.add_option('--time_step', help='Number of minutes per simulation_step', dest='time_step', action='store',
                      type=int, default=60)
    parser.add_option('--simu_step', help='Number of minutes per simulation step', dest='simu_step', action='store',
                      type=int, default=15)
    parser.add_option('--sim', help='Number of replications', dest='nb_sim', action='store',
                      type=int, default=5)
    parser.add_option('-r', '--training_ratio', help='Ratio of the data to use for the training', dest='training_ratio',
                      action='store', type=float, default=0.8)
    parser.add_option('--tep', help='Duration max of an episode occurrence (in minutes)', dest='tep',
                      action='store', type=int, default=30)
    parser.add_option('--plot', help='Display all the important steps', dest='plot', action='store_true',
                      default=False)
    parser.add_option('--debug', help='Display all the intermediate steps', dest='debug', action='store_true',
                      default=False)

    (options, args) = parser.parse_args(args=args)
    # Mandatory Options
    if options.dataset_name is None:
        print("The name of the Input event log is missing\n")
        parser.print_help()
        exit(-1)

    dataset_name = options.dataset_name
    simu_time_step = dt.timedelta(minutes=options.simu_step)
    adp_time_step = dt.timedelta(minutes=options.time_step)
    nb_replications = options.nb_sim
    static_learning = options.static_learning
    window_days = options.window_days
    Tep = options.tep
    debug = options.debug
    plot = options.plot
    training_ratio = options.training_ratio

    print('#' + 'PARAMETERS'.center(28, ' ') + '#')
    print("Dataset Name : {}".format(dataset_name.upper()))
    print("Tep (mn): {}".format(Tep))
    print("Training ratio : {}".format(training_ratio))
    print("Static Learning (no time evolution) : {}".format(static_learning))
    if not static_learning:
        print("Time Window Duration : {} days".format(options.window_days))
    print("Simulation time step (mn) : {}".format(int(simu_time_step.total_seconds() / 60)))
    print("ADP time step (mn) : {}".format(int(adp_time_step.total_seconds() / 60)))
    print("Number of replications : {}".format(nb_replications))
    print("Display Mode : {}".format(plot))
    print("Debug Mode: {}".format(debug))

    dataset = Utils.pick_dataset(dataset_name, nb_days=-1)

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime()

    # Compute the number of periods

    nb_days = math.floor((end_date - start_date).total_seconds() / period.total_seconds())
    # training_days = int(math.floor(nb_days*0.9)) # 90% for training, 10% for test
    training_days = int(nb_days * training_ratio)

    testing_days = nb_days - training_days + 5  # Extra days just in case

    output = "./output/{}/Simulation/{}_tw_{}/".format(dataset_name, 'STATIC' if static_learning else 'DYNAMIC',
                                                       options.window_days)

    # Create the folder if it does not exist yet
    if not os.path.exists(os.path.dirname(output)):
        try:
            os.makedirs(os.path.dirname(output))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    training_start_date = start_date
    training_end_date = start_date + dt.timedelta(days=training_days)

    simulation_start_date = training_end_date
    simulation_duration = dt.timedelta(days=testing_days)

    training_dataset = dataset[(dataset.date >= training_start_date) & (dataset.date < training_end_date)].copy()

    #############
    # LEARNING  #
    #############

    if static_learning:
        print('##############################')
        print("#      STATIC LEARNING       #")
        print('##############################')
        activity_manager = create_static_activity_manager(dataset_name=dataset_name, dataset=training_dataset,
                                                          period=period, time_step=adp_time_step, output=output,
                                                          Tep=Tep, singles_only=False, debug=debug)
    else:
        print('##############################')
        print("#     DYNAMIC LEARNING       #")
        print('##############################')
        activity_manager = create_dynamic_activity_manager(dataset_name=dataset_name, dataset=training_dataset,
                                                           period=period, time_step=adp_time_step, output=output,
                                                           Tep=Tep, nb_days_per_window=window_days, debug=debug)

        activity_manager.dump_data(output=output + "Activity_Manager/")

        print("## Building forecasting models ... ##")
        activity_manager.build_forecasting_duration(dataset, train_ratio=0.95, nb_periods_to_forecast=testing_days + 5)
        ADP_error_df, duration_error_df = activity_manager.build_forecasting_models(train_ratio=0.95,
                                                                                    nb_periods_to_forecast=testing_days + 5,
                                                                                    display=plot, debug=debug)

        ADP_error_df.to_csv(output + '/../ADP_Forecasting_Models_Errors.csv', sep=';', index=False)
        duration_error_df.to_csv(output + '/../Duration_Forecasting_Models_Errors.csv', sep=';', index=False)

    # dump Activity Manager
    pickle.dump(activity_manager, open(output + '{}_Activity_Manager.pkl'.format('STATIC' if static_learning
                                                                                 else 'DYNAMIC'), 'wb'))

    ###############
    # SIMULATION  #
    ###############

    #
    for replication in range(nb_replications):

        time_start = t.process_time()

        print('###################################')
        print('# Simulation replication N°{}     #'.format(replication + 1))
        print('###################################')

        next_time_window_id = activity_manager.last_time_window_id

        if not static_learning:
            next_time_window_id += 1

        simulated_dataset = activity_manager.simulate(start_date=simulation_start_date,
                                                      end_date=simulation_start_date + simulation_duration,
                                                      idle_duration=simu_time_step, time_window_id=next_time_window_id)

        filename = output + "dataset_simulation_rep_{}.csv".format(replication + 1)

        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        simulated_dataset.to_csv(filename, index=False, sep=';')
        elapsed_time = dt.timedelta(seconds=round(t.process_time() - time_start, 1))
        print("Time elapsed for the simulation : {}".format(elapsed_time))


def create_static_activity_manager(dataset_name, dataset, period, time_step, output, Tep, singles_only=False,
                                   debug=False):
    """
    Generate Activities/Macro-Activities from the input event log
    :param dataset_name: Name of the event log
    :param dataset: Input event log
    :param period: periodicity
    :param simu_time_step: simulation time step
    :param output: Output folder for the simulation
    :param Tep: Duration max of an episode occurrence
    :param debug:
    :return:
    """

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime()

    nb_days = math.floor((end_date - start_date).total_seconds() / period.total_seconds())

    activity_manager = ActivityManager.ActivityManager(name=dataset_name, period=period, time_step=time_step,
                                                       tep=Tep, window_size=nb_days, dynamic=False)

    print("Mining for macro-activities...")

    all_macro_activities = Pattern_Mining.Extract_Macro_Activities.extract_macro_activities(dataset=dataset,
                                                                                            support_min=nb_days / 2,
                                                                                            tep=Tep, verbose=False,
                                                                                            singles_only=singles_only)
    for episode, (episode_occurrences, events) in all_macro_activities.items():
        activity_manager.update(episode=episode, occurrences=episode_occurrences, events=events, display=debug)

    log_filename = output + "/parameters.txt"

    with open(log_filename, 'w+') as file:
        file.write("Parameters :\n")
        file.write("STATIC LEARNING (no time evolution)")
        file.write("Periodicity : {}\n".format(period))
        file.write("Macro-Activities Activated : {}\n".format(True))
        file.write("Simulation time step : {} min\n".format(time_step.total_seconds() / 60))
        file.write("Tep : {}\n".format(Tep))

    pickle.dump(activity_manager, open(output + "/Static_Activity_Manager.pkl", 'wb'))
    print('Activity Manager Built & Ready!!')

    return activity_manager


def create_dynamic_activity_manager(dataset_name, dataset, period, time_step, output, Tep, nb_days_per_window,
                                    debug=False):
    """
    Generate Dynamic Macro-Activities from the input event log
    :param dataset_name: Name of the log_dataset
    :param dataset:
    :param period:
    :param time_step:
    :param output:
    :param Tep:
    :param nb_days_per_window: Number of days in a time window
    :param display:
    :return:
    """

    nb_days = int((dataset.date.max().to_pydatetime() - dataset.date.min().to_pydatetime()) / period)

    obs_ratio = 0.25
    max_no_news = 5
    activity_manager = ActivityManager.ActivityManager(name=dataset_name, period=period, time_step=time_step,
                                                       tep=Tep, window_size=nb_days_per_window, max_no_news=max_no_news,
                                                       dynamic=True)

    time_window_duration = dt.timedelta(days=nb_days_per_window)
    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime() - time_window_duration

    support_min = nb_days_per_window

    nb_processes = 2 * mp.cpu_count()  # For parallel computing
    monitoring_start_time = t.time()

    nb_tw = math.floor((end_date - start_date) / period)  # Number of time windows available

    # For each Time Windows Log, we extract the macro-activities and update them
    #############################################################################

    # Arguments for the "extract_tw_macro_activities" method
    args = [(dataset, support_min, Tep, period, time_window_duration, tw_id) for tw_id in range(nb_tw)]

    nb_new_episodes = []
    with mp.Pool(processes=nb_processes) as pool:
        # return tw_id, tw_macro_activities
        all_time_windows_macro_activities = pool.starmap(
            Pattern_Mining.Extract_Macro_Activities.extract_tw_macro_activities, args)

        for result in all_time_windows_macro_activities:
            tw_id = result[0]
            macro_activities = result[1]

            for episode, (episode_occurrences, events) in macro_activities.items():
                # Update of the macro-activity if it exist OR creation if not
                activity_manager.update(episode=episode, occurrences=episode_occurrences, events=events,
                                        time_window_id=tw_id, display=debug)
            print('Activities updates on Time Window {}/{} Done !'.format(tw_id + 1, nb_tw))

            nb_new_episodes.append(len(activity_manager.discovered_episodes))

    # plt.plot(nb_new_episodes)
    # plt.title('Nombre de macro-activités découvert')
    # plt.xlabel('ID Fenêtre temporelle')
    # plt.ylabel('Nombre de macro-activités')
    # plt.show()

    print("Final Macro-Activities List ready")

    elapsed_time = dt.timedelta(seconds=round(t.time() - monitoring_start_time, 1))

    print("###############################")
    print("Time Windows Screening Time : {}".format(elapsed_time))

    pickle.dump(activity_manager, open(output + "/Dynamic_Activity_Manager.pkl", 'wb'))
    print('Activity Manager Built & Ready!!')

    return activity_manager


if __name__ == '__main__':
    main(sys.argv[1:])
