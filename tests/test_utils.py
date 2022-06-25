import pandas as pd
import numpy as np
from xlogit.utils import wide_to_long
from xlogit.utils import lrtest
from xlogit import MixedLogit


dfw = pd.DataFrame({'id': [1, 2, 3, 4, 5],
                    'time_car': [1, 1, 1, 1, 1],
                    'time_bus': [2, 2, 2, 2, 2],
                    'cost_bus': [3, 3, 3, 3, 3],
                    'income': [9, 8, 7, 6, 5],
                    'age': [.6, .5, .4, .3, .2],
                    'y': ['bus', 'bus', 'bus', 'car', 'car']})

def test_wide_to_long():
    """
    Ensures a pandas dataframe is properly converted from wide to long format
    """
    expec = pd.DataFrame({'id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                          'alt': ['car', 'bus', 'car', 'bus', 'car', 'bus',
                                  'car', 'bus', 'car', 'bus'],
                          'time': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                          'cost': [0, 3, 0, 3, 0, 3, 0, 3, 0, 3],
                          'income': [9, 9, 8, 8, 7, 7, 6, 6, 5, 5],
                          'age': [.6, .6, .5, .5, .4, .4, .3, .3, .2, .2],
                          'y': ['bus', 'bus', 'bus', 'bus', 'bus', 'bus',
                                'car', 'car', 'car', 'car']})
    dfl = wide_to_long(dfw, id_col="id", alt_list=["car", "bus"],
                       alt_name="alt", varying=["time", "cost"], empty_val=0)
    assert dfl.equals(expec)


def test_lrtest():
    """
    Ensures a correct result of the lrtest. The comparison values were 
    obtained from comparison with lrtest in R's lmtest package
    """
    general = MixedLogit()
    general.loglikelihood = 1312    
    restricted = MixedLogit()    
    restricted.loglikelihood = -1305    
    general.loglikelihood = -1312    
    general.coeff_ = np.zeros(4)    
    restricted.coeff_ = np.zeros(2)
    
    obtained = lrtest(general, restricted)
    expected = {'pval': 0.0009118819655545164, 'chisq': 14, 'degfree': 2}