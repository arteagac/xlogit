import pandas as pd
from xlogit.utils import wide_to_long


dfw = pd.DataFrame({'id': [1, 2, 3, 4, 5],
                    'time_car': [1, 1, 1, 1, 1],
                    'time_bus': [2, 2, 2, 2, 2],
                    'cost_bus': [3, 3, 3, 3, 3],
                    'income': [9, 8, 7, 6, 5],
                    'age': [.6, .5, .4, .3, .2],
                    'y': ['bus', 'bus', 'bus', 'car', 'car']})

def test_wide_to_long():
    expec = pd.DataFrame({'id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                          'alt': ['bus', 'car', 'bus', 'car', 'bus', 'car',
                                  'bus', 'car', 'bus', 'car'],
                          'time': [2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
                          'cost': [3, 0, 3, 0, 3, 0, 3, 0, 3, 0],
                          'income': [9, 9, 8, 8, 7, 7, 6, 6, 5, 5],
                          'age': [.6, .6, .5, .5, .4, .4, .3, .3, .2, .2],
                          'y': ['bus', 'bus', 'bus', 'bus', 'bus', 'bus',
                                'car', 'car', 'car', 'car']})
    dfl = wide_to_long(dfw, id_col="id", alt_list=["car", "bus"],
                       alt_name="alt", varying=["time", "cost"], empty_val=0)
    assert dfl.equals(expec)