# Facebook Prophet example
import seaborn as sns
from fbprophet import Prophet

from Utils import *

sns.set_style('darkgrid')


def main():
    activity = 'sleeping'
    raw_dataset = pd.read_csv('./data/{}_dataset.csv'.format(activity))

    period = dt.timedelta(days=1)
    nb_feat = len(raw_dataset.columns)
    t_step = period / nb_feat

    dataset = raw_dataset.values.flatten()

    start_date = dt.datetime.now().date()
    start_date = dt.datetime.combine(start_date, dt.datetime.min.time())

    plt.plot(dataset)
    plt.show()

    date_range = [start_date + i * t_step for i in range(len(dataset))]

    dataset = pd.DataFrame(list(zip(date_range, dataset)), columns=['ds', 'y'])

    train_ratio = 0.9
    n_train = int(len(dataset) * train_ratio)
    train_dataset = dataset[:n_train]
    test_dataset = dataset[n_train:]

    model = Prophet(interval_width=0.95, daily_seasonality=True)

    model.fit(train_dataset)

    freq = '{}S'.format(int(t_step.total_seconds()))
    future = model.make_future_dataframe(periods=len(test_dataset), freq=freq)
    forecast = model.predict(future)

    # df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')

    ax = sns.lineplot(x="ds", y="y", data=train_dataset, label='Train')
    sns.lineplot(x="ds", y='yhat', data=forecast, label='Forecast', ax=ax)
    sns.lineplot(x="ds", y='y', data=test_dataset, label='Test', ax=ax)

    # model.plot(forecast)
    model.plot_components(forecast)
    plt.show()


if __name__ == '__main__':
    main()
