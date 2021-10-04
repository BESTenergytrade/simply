"""
JuH: The script "util-py" defines a normal distribution for the pv generation.
"""

import numpy as np
import pandas as pd


def gaussian_pv(x, mean, std):
    """
    JuH: preparation for the pv generation in form of a shifted normal distribution
    Parameters
    ----------
    x
    mean
    std

    Returns
    -------

    """
    return np.exp(-((x - mean) ** 2) / (2 * std ** 2))


def summerize_actor_trading(sc):
    return (
        pd.DataFrame.from_dict([a.traded for a in sc.actors])
        .unstack()
        .apply(pd.Series)
        .rename({0: "energy", 1: "avg_price"}, axis=1)
    )
