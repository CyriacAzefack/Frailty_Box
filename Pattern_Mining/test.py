import datetime as dt
import operator
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from DES import Activity

sns.set_style('darkgrid')

def main():
    dataset_name = 'hh101'
    t_step = dt.timedelta(minutes=60)
    period = dt.timedelta(days=1)
    tep = 30

    all_macro_activities = pickle.load(open('../output/{}/all_macro_activities'.format(dataset_name), 'rb'))

    all_macro_activities = dict(sorted(all_macro_activities.items(), key=operator.itemgetter(0)))

    activityManager = Activity.ActivityObjectManager(name=dataset_name, period=period, time_step=t_step, tep=tep)

    for tw_id, macro_activities in all_macro_activities.items():

        print('UPDATE for Time Window {}!!'.format(tw_id))
        for episode, df_tuple in macro_activities.items():
            # if ('eat_breakfast' in episode) and ('cook_breakfast' in episode) :
            episode_occurrences = df_tuple[0]
            events = df_tuple[1]

            activityManager.update(episode=episode, occurrences=episode_occurrences, events=events,
                                   time_window_id=tw_id)

    activityManager.build_forecasting_models(train_ratio=0.8, method=Activity.MacroActivity.SARIMAX, display=True)
    

def plot_episode_discovery(all_macro_activities):
    """
    Display the evolution of newly discovered macro-activities
    :param all_macro_activities:
    :return:
    """

    all_macro_activities = dict(sorted(all_macro_activities.items(), key=operator.itemgetter(0)))

    known_episodes = []
    new_episodes_evol = []

    for tw_id, macro_activities in all_macro_activities.items():
        # print('ID {} : {}'.format(tw_id, macro_activities.keys()))

        episodes = list(macro_activities.keys())

        episodes = [frozenset(e) for e in episodes]
        old_episodes = list(set(episodes) & set(known_episodes))
        new_episodes = set(episodes) - set(old_episodes)
        nb_new_episodes = len(new_episodes)

        for episode in new_episodes:
            known_episodes.append(episode)

        new_episodes_evol.append(nb_new_episodes)

    plt.plot(np.arange(len(new_episodes_evol)), new_episodes_evol)
    plt.title('Number of macro discovered')
    plt.xlabel('Time Windows')
    plt.ylabel('nb episodes')
    plt.show()
if __name__ == "__main__":
    main()