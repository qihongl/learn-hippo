import numpy as np
import pandas as pd


def one_hot_to_int(one_hot_vector):
    """convert a onehot vector (or zero-hot vector) to its index representation
    if zero-hot, then return np.nan

    Parameters
    ----------
    one_hot_vector : 1d np.array
        an one hot vector

    Returns
    -------
    index
        i s.t. one_hot_vector[i] == 1

    """
    one_hot_index = np.where(one_hot_vector)[0]
    n_ones = len(one_hot_index)
    if n_ones == 1:
        return int(one_hot_index)
    elif n_ones == 0:
        return np.nan
    else:
        raise ValueError(f'Invalid one-hot vector: {one_hot_vector}')


def prop_true(bool_array, axis=0):
    """compute the proportion of truth values along a axis

    Parameters
    ----------
    bool_array : nd array
        boolean array
    axis : int
        array axis to sum over

    Returns
    -------
    (n-1)d array
        % true along input axis

    """
    n = np.shape(bool_array)[axis]
    return np.sum(bool_array, axis=axis) / n


def make_df(data_dict):
    """
    convert a data dictionary to a dataframe
    the data dict must be in the form of {'condition_name': 1-d data array}

    Parameters
    ----------
    data_dict : dict
        a data dictionary

    Returns
    -------
    pd.DataFrame
        a data frame: col1 = value; col2 = condition labels

    """
    # get sample size
    n_data = dict(zip(
        data_dict.keys(), [len(data) for data in data_dict.values()]
    ))
    # get condition vector
    cond = np.concatenate(
        [[cond_name] * n_data[cond_name] for cond_name in data_dict.keys()]
    )
    # get data value vector
    data = np.concatenate([cond_data for cond_data in data_dict.values()])
    # combine to form a df
    df = pd.DataFrame({'Value': data, 'Condition': cond})
    return df
