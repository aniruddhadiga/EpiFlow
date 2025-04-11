import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def optimize_savgol_filter(y_noisy, window_range=(7, 15, 2), max_poly=3):
    """
    Finds the best Savitzky-Golay filter parameters based on RMSE.

    Parameters:
    - y_noisy: np.array, the noisy input signal
    - window_range: tuple, (start, stop, step) for odd window sizes
    - max_poly: int, maximum polynomial order to test

    Returns:
    - best_params: tuple (window_length, polyorder)
    - best_filtered: np.array, filtered signal with best params
    - cv_dict: dictionary of mean squared error
    """
    best_mse = float('inf')
    best_params = (0, 0)
    best_filtered = None
    cv_dict={}
    for window in range(*window_range):
        cv_dict[window]={}
        for order in range(1, min(window, max_poly + 1)):
            y_filtered = savgol_filter(y_noisy, window_length=window, polyorder=order)
            mse = mean_squared_error(y_noisy, y_filtered)
            cv_dict[window][order]=mse
            if mse < best_mse:
                best_mse = mse
                best_params = (window, order)
                best_filtered = y_filtered

   

    print(f"Best params: window_length={best_params[0]}, polyorder={best_params[1]}, RMSE={best_mse:.4f}")
    return best_filtered, best_params, cv_dict




def denoise_VAL(df, optimize=False):
    all_sig=pd.DataFrame()
    cols=['geo_value','time_value','geo_res','viral_activity_level']
    for loc in df.geo_value.unique():
        rdf=df[df.geo_value==loc][cols]
        sig=rdf[['time_value','viral_activity_level']].set_index('time_value')
        sig=sig.ffill()
        y=sig.dropna().values
        ind=sig.dropna().index
        y=y.reshape(-1)
        if optimize:
            yf,best_params,_=optimize_savgol_filter(y)
        else:
            yf=savgol_filter(y,7,2)
        sig.loc[ind,'dn_viral_activity_level']=yf
        sig.loc[:,'geo_value']=loc
        all_sig=pd.concat([all_sig, sig.reset_index()])
    return all_sig