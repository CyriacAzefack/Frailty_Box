from Pattern_Mining import Candidate_Study
from Utils import *

DECIMAL_PRECISION = 1000


def main():
    """
    Check the distribution of frequency vs periodicity
    :return:
    """

    dataset_name = 'aruba'
    dataset = pick_dataset(dataset_name)
    period = dt.timedelta(days=1)
    t_step = 60  # in minutes

    activity = 'meal_preparation'

    dataset = dataset[dataset.label == activity]

    dataset['relative_date'] = dataset.date.apply(lambda x: Candidate_Study.modulo_datetime(x.to_pydatetime(), period))

    time_window_duration = dt.timedelta(days=30)

    start_date = dataset.date.min().to_pydatetime()
    end_date = dataset.date.max().to_pydatetime() - time_window_duration

    window_start_date = start_date

    min_bin = 0
    max_bin = 24 * 3600  # 24 hours
    t_step *= 60  # convert in seconds

    bins = int(max_bin / t_step)

    features = np.arange(min_bin, max_bin, t_step)
    features = ["Ts_{}".format(i) for i in range(len(features))]

    built_dataset = pd.DataFrame(columns=features)

    i = 0
    while window_start_date < end_date:
        window_end_date = window_start_date + time_window_duration

        # extract occurrences time
        ts_array = np.asarray(dataset.loc[(dataset.date >= window_start_date) & (dataset.date < window_end_date),
                                          "relative_date"].values)

        # Compute the count_histogram

        # bin_width = bin_minutes * 60

        # bins = np.linspace(min_val, max_val, int(max_val / bin_minutes))

        histo, _ = np.histogram(ts_array, bins=bins, range=(min_bin, max_bin), density=False)

        built_dataset.loc[i] = histo

        window_start_date += period
        i += 1

    # built_dataset.to_csv('{}_dataset.csv'.format(activity), period_ts_index=False)

    for feat in features:
        plt.plot(built_dataset.index, built_dataset[feat], label=feat)

    plt.xlabel('Time Windows')
    plt.ylabel('Occurrences count')
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
