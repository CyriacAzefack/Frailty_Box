import operator
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('darkgrid')

def main():
    dataset_name = 'aruba'

    all_macro_activities = pickle.load(open('../output/{}/all_macro_activities'.format(dataset_name), 'rb'))

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