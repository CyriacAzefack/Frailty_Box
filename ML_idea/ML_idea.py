import datetime as dt
import itertools
import time as t
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from xED.Pattern_Discovery import pick_dataset


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


dataset_name = 'aruba'
dataset = pick_dataset(dataset_name)

## Features to extract :
#     * 'lastEventHour' : Hour of the day when the current event occurred
#     * 'lastEventSeconds' : Time in seconds since the beginning of the day for the current event
#     * 'timeStamp' : time since the beginning of the day (seconds) normalized by the total number of seconds in a day
#     * 'windowDuration' : Duration of the window in seconds (last 5 events for instance)
#     * 'timeSinceLastEvent' : Time since the previous event (seconds)
#     ** 'prevDominantSensor1' : The most frequent label ID in the previous window
#     ** 'prevDominantSensor2' : The most frequent label ID in the window before that'
#     * 'lastLabelID' : The label ID of the current event
#     * 'lastEventDuration' : duration of the current event
#     * 'lag [all labels id]' : timestamp feature value for the
#
## Targets:
#    * 'timeForNextEvent' : time in seconds fo the next event
#    * 'nextLabelID' : label ID for the next event

labels = dataset.label.unique()
labels_df = pd.DataFrame(labels, columns=['label'])
labels_df.sort_values(by=['label'], ascending=True, inplace=True)
labels_df['label_id'] = np.arange(len(labels))

# Build targets
start_time = t.process_time()
dataset['next_label'] = dataset['label'].shift(-1)
dataset = pd.merge(dataset, labels_df, how='left', left_on='next_label', right_on='label', suffixes=('', '_out'))
dataset.rename(index=str, columns={"label_id": "next_label_id"}, inplace=True)
dataset.drop(columns=['label_out'])
print("Target 'next_label_id' done")

dataset['time_for_next_event'] = (dataset['date'].shift(-1) - dataset['date']).apply(lambda x: x.total_seconds())
print("Target 'time_for_next_event' done")

elapsed_time = dt.timedelta(seconds=round(t.process_time() - start_time, 1))
print("Time to build targets : {}".format(elapsed_time))

# Build Features
start_time = t.process_time()

dataset['day_date'] = dataset['date'].dt.date.apply(lambda x: dt.datetime.combine(x, dt.datetime.min.time()))

# lastEventHour
dataset['last_event_hour'] = dataset['date'].apply(lambda x: x.hour)
print("Feature 'LastEventHour' done")

# lastEventSeconds
dataset['last_event_seconds'] = dataset['date'] - dataset['day_date']
dataset['last_event_seconds'] = dataset['last_event_seconds'].apply(lambda x: x.total_seconds())
print("Feature 'last_event_seconds' done")

# timeStamp
dataset['timestamp'] = dataset['last_event_seconds'] / (24 * 3600)
print("Feature 'timestamp' done")

# windowDuration
window_size = 10

# timeSinceLastEvent
dataset['time_since_last_event'] = (dataset['date'] - dataset['date'].shift(1)).apply(lambda x: x.total_seconds())
print("Feature 'time_since_last_event' done")

# lastLabelID
dataset['last_label'] = dataset['label']
dataset = pd.merge(dataset, labels_df, how='left', left_on='last_label', right_on='label', suffixes=('', '_out'))
dataset.rename(index=str, columns={"label_id": "last_label_id"}, inplace=True)
dataset.drop(columns=['label_out'])
print("Feature 'last_label_id' done")

# lastEventDuration
dataset['last_event_duration'] = (dataset['end_date'] - dataset['date']).apply(lambda x: x.total_seconds())

print("Feature 'last_event_duration' done")

# lag

# def time_since_last_label(row, label):
#     last_label_occ = dataset[(dataset.label == label) & (dataset.date < row.date)].date.max().to_pydatetime()
#     return (row.date - last_label_occ).total_seconds()
#
# for label in labels :
#     dataset[label+'_lag'] = dataset.apply(time_since_last_label, axis=1, args=(label,))
#
# labels_feat = [label+'_lag' for label in labels]

elapsed_time = dt.timedelta(seconds=round(t.process_time() - start_time, 1))
print("Time to build Features : {}".format(elapsed_time))

dataset = dataset.dropna()

features = ['last_event_hour', 'last_event_seconds', 'timestamp', 'time_since_last_event', 'last_label_id',
            'last_event_duration']  # *labels_feat]

target = ['next_label_id']  # 'time_for_next_event']

## SPLIT
feat_train, feat_test, target_train, target_test = train_test_split(dataset[features], dataset[target], test_size=0.33,
                                                                    random_state=42)

## TRAINING
clf = RandomForestClassifier(n_estimators=20)


# clf.fit(feat_train, target_train)
#
# scores = cross_val_score(clf, feat_test, target_test, cv=5)
# print('Score Mean : {}'.format(scores.mean()))
#
# target_predict = clf.predict(feat_test)
# f1 = f1_score(target_test, target_predict, average=None)
# print('F1 score : {}'.format(f1.mean()))
#
# cnf_matrix = confusion_matrix(target_test, target_predict)
# np.set_printoptions(precision=2)
#
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
#                       title='Normalized confusion matrix')
#
# plt.show()


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, len(features)),
              "min_samples_split": sp_randint(2, len(features)),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)

start = t.time()
random_search.fit(feat_train, target_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((t.time() - start), n_iter_search))
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 5],
              "min_samples_split": [2, 3, 5],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
start = t.time()
grid_search.fit(feat_train, target_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (t.time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
