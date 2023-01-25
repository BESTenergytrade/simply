import numpy as np
import pandas as pd


def gaussian_pv(ts_hour, std):
    # One day (24h) time series with peak at noon (12h)
    # and a gaussian curve defined by standard deviation
    # todo fix to 24 hours
    x = np.linspace(0, 24 * ts_hour-1, 24 * ts_hour)
    mean = 12 * ts_hour
    std = std * ts_hour
    return np.exp(-((x - mean) ** 2) / (2 * std ** 2))


def daily(df, daily_ts=24):
    for i in range(0, len(df), daily_ts):
        yield df.iloc[i:i + daily_ts]


def summerize_actor_trading(sc):
    # Check if at least one trade has happened
    empty = True
    for i in [a.traded for a in sc.actors]:
        if len(i) != 0:
            empty = False
    if not empty:
        return (
            pd.DataFrame.from_dict([a.traded for a in sc.actors])
            .unstack()
            .apply(pd.Series)
            .rename({0: "energy", 1: "avg_price"}, axis=1)
        )


def get_all_data(df, col="pv"):
    """
    Select all columns 'col' at subcolumn level of the actors DataFrame.

    :param df: actor DataFrame with multi-column-index (actor_col, assets_col)
    :param col: selected assets_col
    :return: DataFrame with single column level comprising all columns of equal sub-column name col
    """
    return df.iloc[:, df.columns.get_level_values(1) == col]
