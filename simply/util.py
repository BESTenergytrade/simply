"""
JuH: The script "util.py" defines a normal distribution for the pv generation.
bucket for helper functions that are not really tied to the market."""

import numpy as np
import pandas as pd


def gaussian_pv(x, mean, std):
    """
    JuH: preparation for the pv generation in form of a shifted normal distribution
    Returns

    Parameters
    ----------
    x
    mean
    std
    -------
    Returns: value on gaussian bell curve for a given mean and standard deviation.
        factors missing?
    """
    return np.exp(-((x - mean) ** 2) / (2 * std ** 2))


def summerize_actor_trading(sc):
    """StS: returns all trades for a given scenario.
    why is it not a member function of Scenario?"""
    return (
        pd.DataFrame.from_dict([a.traded for a in sc.actors])
        .unstack()
        .apply(pd.Series)
        .rename({0: "energy", 1: "avg_price"}, axis=1)
    )
