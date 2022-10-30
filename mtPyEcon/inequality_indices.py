import numpy as np
import pandas as pd

def lorenz_curve(fx, x_dist):
    # Check the type of each input
    if type(fx) is list:
        fx = np.array(fx)
    elif type(fx) is pd.core.series.Series:
        fx = fx.to_numpy()
    if type(x_dist) is list:
        x_dist = np.array(x_dist)
    elif type(x_dist) is pd.core.series.Series:
        x_dist = x_dist.to_numpy()
    # Check if the sum of the distribution is unity
    if np.sum(x_dist) != 1:
        x_dist = x_dist / np.sum(x_dist)
    # Sort the data
    sorted_idx = fx.argsort()
    x_dist = x_dist[sorted_idx]    
    fx.sort()
    # Calculate the cumulative share in aggregate earnings
    fx_contrib = fx * x_dist
    cum_fx_share = np.cumsum(fx_contrib)/np.sum(fx_contrib)
    cum_fx_share = np.insert(cum_fx_share, 0 , 0.0)
    # Calculate the cumulative share in total samples
    cum_N_share = np.cumsum(x_dist)/np.sum(x_dist)
    cum_N_share = np.insert(cum_N_share, 0 , 0.0)
    # Combine the two series into an array
    lorenz_curve = np.array([cum_N_share, cum_fx_share])
    return lorenz_curve

def gini_index(fx, x_dist):
    # Prepare the Lorenz curve
    lc = lorenz_curve(fx, x_dist)
    # Numerical integration for the area below the Lorenz curve
    gini_contrib = [(lc[1, i] + lc[1, i+1]) * (lc[0, i+1] - lc[0, i]) * 0.5
                    for i in range(np.size(lc, 1) - 1)]
    # Calculate the Gini coefficient
    gini_index = (0.5 -  np.sum(gini_contrib)) / 0.5
    return gini_index