import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gaussian_pv(ts_hour, std):
    # One day (24h) time series with peak at noon (12h)
    # and a gaussian curve defined by standard deviation
    x = np.linspace(0, 24 * ts_hour, 24 * ts_hour)
    mean = 12 * ts_hour
    return np.exp(-((x - mean) ** 2) / (2 * std ** 2))


def daily(df, daily_ts=24):
    for i in range(0, len(df), daily_ts):
        yield df.iloc[i:i + daily_ts]


def summerize_actor_trading(sc):
    return (
        pd.DataFrame.from_dict([a.traded for a in sc.actors])
        .unstack()
        .apply(pd.Series)
        .rename({0: "energy", 1: "avg_price"}, axis=1)
    )


def get_all_data(df, col="pv"):
    return df.iloc[:, df.columns.get_level_values(1) == col]


def concat_actor_data(sc):
    data = [a.data for a in sc.actors]
    return pd.concat(data, keys=range(len(sc.actors)), axis=1)


def plot_actor_data(sc):
    actor_data = concat_actor_data(sc)
    fig, ax = plt.subplots(3)
    ax[0].set_title("PV")
    ax[1].set_title("Load")
    ax[2].set_title("Sum")
    get_all_data(actor_data, "pv").plot(ax=ax[0], legend=False)
    get_all_data(actor_data, "load").plot(ax=ax[1], legend=False)
    get_all_data(actor_data, "pv").sum(axis=1).plot(ax=ax[2])
    get_all_data(actor_data, "load").sum(axis=1).plot(ax=ax[2])
    ax[2].legend(["pv", "load"])
