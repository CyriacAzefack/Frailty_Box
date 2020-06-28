#######################
# FEATURES EXTRACTION #
#######################

# Implementation of all the features extraction method for time windows
import spm1d
from scipy import stats
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric.kde import KDEUnivariate

import Data_Drift.Drift_Detector as Drift_Detector
from Utils import *


def main():
    data_name = 'KA'

    data = pick_dataset(data_name)
    # label = "bed_toilet_transition"
    time_window_duration = dt.timedelta(days=7)

    labels = data.label.unique()

    ##########################
    ## Pre-Treatment of data #
    ##########################

    data['day_date'] = data['date'].dt.date.apply(lambda x: dt.datetime.combine(x, dt.datetime.min.time()))
    data['timestamp'] = (data['date'] - data['day_date']).apply(lambda x: x.total_seconds())

    data['duration'] = (data['end_date'] - data['date']).apply(lambda x: x.total_seconds() / 3600)

    #############################
    ## Creation of time windows #
    #############################

    time_windows_data = Drift_Detector.create_time_windows(data, time_window_duration)

    nb_windows = len(time_windows_data)

    ######################
    ## Similarity Matrix #
    ######################

    similarity_matrix = np.empty((nb_windows, nb_windows))

    for i in range(nb_windows):
        tw_data_A = time_windows_data[i]
        for j in range(nb_windows):
            tw_data_B = time_windows_data[j]
            similarities = []
            # TODO : Add weights for the different type of similarity
            for label in labels:
                # 1- Occurrence time similarity
                arrayA = tw_data_A[tw_data_A.label == label].timestamp.values
                arrayB = tw_data_B[tw_data_B.label == label].timestamp.values

                occ_time_similarity = ks_similarity(arrayA, arrayB)

                arrayA = tw_data_A[tw_data_A.label == label].duration.values
                arrayB = tw_data_B[tw_data_B.label == label].duration.values

                duration_similarity = ks_similarity(arrayA, arrayB)
                similarity = (duration_similarity + occ_time_similarity) / 2
                similarities.append(similarity)
            similarity_matrix[i][j] = np.mean(similarities)


def ks_similarity(arrayA, arrayB):
    """
    Compute the similarity between 2 time windows event sequence using a Kolmogorov–Smirnov test.
    Null hypothesis : The two data array come from the same distribution
    :param array_A:
    :param arrayB:
    :return:
    """

    if (len(arrayA) == 0) and (len(arrayB) == 0):
        return 1
    elif (len(arrayA) == 0) or (len(arrayB) == 0):
        return 0

    _, p_val = stats.ks_2samp(arrayA, arrayB)

    if np.isnan(p_val):
        return 0

    return p_val


def activities_features(data, activity_labels):
    """
    Comptue the mean_time, std_time and nb_occ for all labels
    :param data:
    :param activity_labels:
    :return: all the features
    """
    features = {}

    # nb_days = np.ceil((data.end_date.max() - data.date.min()) / dt.timedelta(days=1))

    for label in activity_labels:
        label_data = data[data.label == label]
        ## 1 - Features on activities occurrence time
        label_occ_time = label_data.timestamp.values
        features[label + "_mean_occ_time"] = [np.mean(label_occ_time)]
        features[label + "_std_occ_time"] = [np.std(label_occ_time)]

        ## 2 - Feature on the duration of activities
        label_durations = (label_data.end_date - label_data.date).apply(lambda x: x.total_seconds() / 60)
        features[label + "_mean_duration"] = [np.mean(label_durations)]
        features[label + "_std_duration"] = [np.std(label_durations)]

        ## 3 - Feature on the number of occurrences
        features[label + '_nb_occ'] = [len(label_data)]

    return features


def mse(array_A, array_B, bin_width=5 * 60, max_bin=24 * 3600):
    """
    Compute the MSE (Mean Squared Error) of the normalized histograms on the 2 arrays
    :param arrayA:
    :param arrayB:
    :param bins: nb of data points for the kernel density
    :return:
    """

    if (len(array_A) == 0) and (len(array_B) == 0):
        return np.nan

    if len(array_A) == 0:
        array_A = np.zeros_like(array_B)

    if len(array_B) == 0:
        array_B = np.zeros_like(array_A)

    min_bin = 0
    # max_bin = 24 * 3600  # 24hours

    bins = int(max_bin / bin_width)

    # bins = np.linspace(min_val, max_val, int(max_val / bin_minutes))

    hist_A, _ = np.histogram(array_A, bins=bins, range=(min_bin, max_bin), density=False)

    hist_B, _ = np.histogram(array_B, bins=bins, range=(min_bin, max_bin), density=False)

    # bin_min = min(list(arrayA) + list(arrayB))
    # bin_max = max(list(arrayA) + list(arrayB))
    # X_plot = np.linspace(bin_min, bin_max, bins)

    mse = mean_squared_error(hist_A, hist_B)

    return mse


def density_intersection_area(arrayA, arrayB, bins=200):
    """
    Compute the intersection area of the kernel density on the two arrays
    :param arrayA:
    :param arrayB:
    :param bins: nb of data points for the kernel density
    :return:
    """
    if np.array_equal(arrayA, arrayB):
        return 1

    if (len(arrayA) <= 1) or (len(arrayB) <= 1):
        return 0

    kdeA = KDEUnivariate(arrayA)
    kdeA.fit(bw="scott")

    kdeB = KDEUnivariate(arrayB)
    kdeB.fit(bw='scott')

    bin_min = min(list(kdeA.support) + list(kdeB.support))
    bin_max = max(list(kdeA.support) + list(kdeB.support))

    X_plot = np.linspace(bin_min, bin_max, bins)

    fitted_A = kdeA.evaluate(X_plot)
    fitted_B = kdeB.evaluate(X_plot)

    intersection = [min(fitted_A[i], fitted_B[i]) for i in range(bins)]

    area = metrics.auc(X_plot, intersection)

    if area == np.nan:
        return 0

    return area


def histogram_intersection(array_A, array_B, bin_width=5 * 60, max_bin=24 * 3600):
    """
    Compute the intersection of 2 histograms
    :param array_A:
    :param array_B:
    :param bin_width: input_width of a bin in seconds
    :param max_bin: maximum value of the bin
    :return:

    """
    if (len(array_A) == 0) and (len(array_B) == 0):
        return 0

    if len(array_A) == 0:
        array_A = np.zeros_like(array_B)

    if len(array_B) == 0:
        array_B = np.zeros_like(array_A)

    # bin_width = bin_minutes * 60
    min_bin = 0
    # max_bin = 24 * 3600  # 24hours

    bins = int(max_bin / bin_width)

    # bins = np.linspace(min_val, max_val, int(max_val / bin_minutes))

    hist_A, _ = np.histogram(array_A, bins=bins, range=(min_bin, max_bin), density=True)

    hist_B, _ = np.histogram(array_B, bins=bins, range=(min_bin, max_bin), density=True)

    area = 0
    for i in range(bins):
        area += min(bin_width * hist_A[i], bin_width * hist_B[i])

    # intersection = np.minimum(hist_A, hist_B)
    # area = bin_minutes * np.sum(intersection)

    return area


def hotelling_test(arrayA, arrayB):
    """
    Compute the similarity between 2 time windows event sequence using a Kolmogorov–Smirnov test.
    Null hypothesis : The two data array come from the same distribution
    :param array_A:
    :param arrayB:
    :return:
    """

    if (len(arrayA) == 0) and (len(arrayB) == 0):
        return 1
    elif (len(arrayA) == 0) or (len(arrayB) == 0):
        return 0
    elif (len(arrayA) == 1) and (len(arrayB) == 1):
        if arrayA[0][0] == arrayB[0][0]:
            return 1
        else:
            return 0

    # print(len(arrayA), len(arrayB))
    T2 = spm1d.stats.hotellings2(arrayA, arrayB)
    T2i = T2.inference(0.05)
    p_val = T2i.p

    if np.isnan(p_val):
        return 0

    return p_val


if __name__ == '__main__':
    main()
