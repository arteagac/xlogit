import numpy as np
from xlogit import MixedLogit

model = MixedLogit()
X = np.array([[5, .1, .2],
              [5, .2, .1],
              [5, .3, .3],
              [6, .1, .1],
              [6, .1, .2],
              [6, .2, .1]])

y = np.array([0, 0, 1, 0, 1, 0])
alternatives = ['car', 'bus', 'train']
varnames = ['income', 'price', 'catch']
isvars = ['income']
asvars = ['price', 'catch']
base_alt = 'bus'
fit_intercept = True
max_iterations = 1000


def test__validate_input():
    pass
