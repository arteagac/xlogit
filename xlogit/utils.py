"""General utilities for xlogit library."""

import numpy as np

def wide_to_long(dataframe, id_col, alt_list, alt_name, varying=None,
                 sep="_", alt_is_prefix=False, empty_val=np.nan):
    """Reshapes pandas DataFrame from wide to long format.

    Parameters
    ----------
    dataframe : pandas DataFrame
        The wide-format DataFrame.

    id_col : str
        Column that uniquely identifies each sample.

    alt_list : list-like
        List of choice alternatives.

    alt_name : str
        Name of the alternatives column in returned dataset.

    varying : list-like
        List of column names that vary across alternatives.

    sep : str, default='_'
        Separator of column names that vary across alternatives.

    avail: array-like, shape (n_samples,), default=None
        Availability of alternatives for the choice situations. One when
        available or zero otherwise.

    alt_is_prefix : bool
        True if alternative is prefix of the variable name or False if it is
        suffix.

    empty_val : int, float or str, default=np.nan
        Value to fill when alternative not available for a certain variable.


    Returns
    -------
    DataFrame in long format.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas installation required for reshaping data")
    
    # Validations
    if any(col in varying for col in dataframe.columns):
        raise ValueError("varying can't be identical to a column name")
    if alt_name in dataframe.columns:
        raise ValueError("alt_name can't be identical to a column name")
    
    # Initialize new dataframe with id and alt columns
    newdf = pd.DataFrame()
    newdf[id_col] = np.repeat(dataframe[id_col].values, len(alt_list))
    newdf[alt_name] = np.tile(alt_list, len(dataframe))
    conc_cols = []
    varying = varying if varying is not None else []
    
    # Reshape columns that vary across alternatives
    patt = "{alt}{sep}{col}" if alt_is_prefix else "{col}{sep}{alt}"
    count_match_patt = 0
    for col in varying:
        series = []
        for alt in alt_list:
            c = patt.format(alt=alt, sep=sep, col=col)
            conc_cols.append(c)
            if c in dataframe.columns:
                series.append(dataframe[c].values)
                count_match_patt += 1
            else:
                series.append(np.repeat(empty_val, len(dataframe)))
        newdf[col] = np.stack(series, axis=1).ravel()
    if count_match_patt == 0 and len(varying) > 0:
        raise ValueError(f"no column matches the pattern {patt}")

    # Reshape columns that do NOT vary across alternatives
    non_varying = [c for c in dataframe.columns if c not in conc_cols+[id_col]]
    for col in non_varying:
        newdf[col] = np.repeat(dataframe[col].values, len(alt_list))
    return newdf.sort_values(by=[id_col, alt_name], ignore_index=True)