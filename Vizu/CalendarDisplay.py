# ----------------------------------------------------------------------------
# Author:  Nicolas P. Rougier
# License: BSD
# ----------------------------------------------------------------------------
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from matplotlib.patches import Polygon


def calmap(ax, title, data, start_date, end_date, nb_clusters):
    ax.tick_params('x', length=0, labelsize="medium", which='major')
    ax.tick_params('y', length=0, labelsize="x-small", which='major')

    # Month borders
    xticks, labels = [], []
    # start = datetime(year, 1, 1).weekday()

    start_year = start_date.year
    year = start_year
    start_month = start_date.month

    # nb_months = math.floor((end_date - start_date)/timedelta(days=30))
    nb_months = relativedelta(end_date, start_date).months + 1

    print(f'Start date : {start_date}')
    print(f"Number of months : {nb_months}")
    print(f'End date : {end_date}')

    start_date_month = datetime(start_year, start_month, 1)

    start_index = datetime(start_year, 1, 1).weekday()

    for month in range(1, nb_months + 1):

        first = start_date_month + relativedelta(months=month - 1)
        # first = datetime(year, month, 1)
        last = first + relativedelta(months=1, days=-1)

        if month == 1:
            first = start_date
            last = start_date_month + relativedelta(months=1, days=-1)

        if month == nb_months:
            last = end_date

        y0 = first.weekday()
        y1 = last.weekday()
        # x0_pass = ((first - start_date_month).days + start_index -1) // 7
        # x1_pass = ((last - start_date_month).days + start_index - 1) // 7

        year_diff = first.year - start_year
        x0 = (int(first.strftime("%j")) + year_diff * 366 + start_index - 1) // 7
        x1 = (int(last.strftime("%j")) + year_diff * 366 + start_index - 1) // 7
        #
        # if first.year != start_year:
        #
        #     x0 = int(first.strftime("%j"))+

        P = [(x0, y0), (x0, 7), (x1, 7),
             (x1, y1 + 1), (x1 + 1, y1 + 1), (x1 + 1, 0),
             (x0 + 1, 0), (x0 + 1, y0)]
        xticks.append(x0 + (x1 - x0 + 1) / 2)
        labels.append(first.strftime("%b"))
        poly = Polygon(P, edgecolor="black", facecolor="None",
                       linewidth=1, zorder=20, clip_on=False)
        ax.add_artist(poly)

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(0.5 + np.arange(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_title(title, weight="semibold")
    #
    # # Clearing first and last day from the data
    # valid = datetime(year, 1, 1).weekday()
    # data[:valid, 0] = np.nan
    # valid = datetime(year, 12, 31).weekday()
    # # data[:,x1+1:] = np.nan
    # data[valid + 1:, x1] = np.nan
    #
    # # Showing data

    ax.imshow(data, extent=[0, 53, 0, 7], zorder=10, vmin=0, vmax=nb_clusters,
              cmap="tab20", origin="lower", alpha=.75, )
    #
    # for (j, i), label in np.ndenumerate(data):
    #     date = datetime(year, 1, 1) + timedelta(days=i*52+j - start)
    #     label = int(date.strftime("%d"))
    #     ax.text(i, j, label, ha='center', va='center')
    #     # ax2.text(i, j, label, ha='center', va='center')


def display_calendar(dataset):
    """
    Display the behavior evolution
    :param dataset:
    :return:
    """

    n_clusters = len(dataset.cluster.unique())

    dataset['year'] = dataset.date.apply(lambda x: x.year)

    # To split years
    years = list(dataset.year.unique())

    fig = plt.figure(figsize=(8, 4.5), dpi=100)

    for year in years:
        year_df = dataset[dataset.year == year]

        current_start_date = year_df.date.min()
        current_end_date = year_df.date.max()

        ax = fig.add_subplot(int(f"{len(years)}1{1 + years.index(year)}"), xlim=[0, 53], ylim=[0, 7], frameon=False,
                             aspect=1)

        # Build the data
        # Clearing first and last day from the data

        data = np.empty((7, 53))
        data[:] = np.nan
        data = data.T.flatten()

        valid = current_start_date.dayofyear + datetime(year, 1, 1).weekday() - 1

        data[valid:valid + len(year_df)] = year_df.cluster.values

        # data[:valid] = np.nan
        # valid = current_end_date.dayofyear + datetime(year, 12, 31).weekday() - (1 if year != max(years) else 0)
        # # data[:,x1+1:] = np.nan
        # data[valid:] = np.nan

        data = data.reshape(53, 7).T

        calmap(ax, title=year, data=data, start_date=current_start_date.to_pydatetime(),
               end_date=current_end_date.to_pydatetime(), nb_clusters=n_clusters)

    plt.tight_layout()
    # plt.savefig("calendar-heatmap.png", dpi=300)
    # plt.savefig("calendar-heatmap.pdf", dpi=600)
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    start_date = datetime(2020, 7, 8)
    end_date = datetime(2022, 11, 19)

    x = pd.date_range(start_date, end_date, freq='1D')

    nb_clusters = 4
    clusters = np.random.randint(nb_clusters, size=len(x))

    df = pd.DataFrame({'date': x, 'cluster': 2})

    display_calendar(df)
