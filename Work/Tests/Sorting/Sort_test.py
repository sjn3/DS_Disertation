
import pandas as pd
import numpy as np
from scipy.stats import truncnorm, truncexpon

def lin_list(stop, start=0, step=1):
    """
    This function will give a linear list of values in a dataframe with index values.
    Inputs are: stop, start=0, step=1
    """
    values = np.arange(start, stop+step, step, np.float)
    return pd.DataFrame(values, columns=['lin_values'])

def norm_list(location, num_points, a_shape=-5, b_shape=5, not_int=False):
    """
    This function will give a normal/gauss distribution of numbers
    """
    values = truncnorm(a=a_shape, b=b_shape, loc=location, scale=location/3.33).rvs(size=num_points)
    
    if num_points <= 700 or not_int:
        print('Due to smaller number of points the output will not be in intergers')
        return pd.DataFrame(values, columns=['norm_values'])
    
    values = np.rint(values).astype(int)
    return pd.DataFrame(values, columns=['norm_values'])

def exp_list(shape, num_points, start=0, stop = 50, not_int=False):
    """
    This function will give an exponential distribution of numbers
    """
    num_points = int(num_points)
    values = truncexpon(b=shape, loc=start, scale=1).rvs(size = num_points)
    
    if (stop >= 50) and not(not_int):
        new_scale = stop/np.max(values)
        values = truncexpon(b=shape, loc=start, scale = new_scale).rvs(size = num_points)
    else:
        print('Due to smaller range the output will not be in intergers')
        new_scale = stop/np.max(values)
        values = truncexpon(b=shape, loc=start, scale = new_scale).rvs(size = num_points)
        return pd.DataFrame(values, columns=['exp_values'])
    
    values = np.rint(values).astype(int)
    
    return pd.DataFrame(values, columns=['exp_values'])
