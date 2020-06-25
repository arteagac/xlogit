import pytest
import numpy as np

from pymlogit.linear import MultinomialModel

model = MultinomialModel()
			   
X = np.array([[5, .1, .2],
			  [5, .2, .1],
			  [5, .3, .3],
			  [6, .1, .1],
			  [6, .1, .2],
			  [6, .2, .1]])

y = np.array([0,0,1,0,1,0])

alternatives = ['car','bus','train']
varnames = ['income','price','catch']
isvars = ['income']
asvars = ['price','catch']
base_alt = 'bus'
fit_intercept = True
max_iterations = 2000

def test__validate_input():
	with pytest.raises(ValueError):
		model._validate_inputs(np.array([]), y, alternatives = alternatives, varnames = varnames, asvars = asvars, isvars = isvars,
			base_alt = base_alt, fit_intercept = fit_intercept, max_iterations = max_iterations)

	with pytest.raises(ValueError):
		model._validate_inputs(X, y, alternatives = [], varnames = varnames, asvars = asvars, isvars = isvars,
			base_alt = base_alt, fit_intercept = fit_intercept, max_iterations = max_iterations)

	with pytest.raises(ValueError): #without alternatives
		model._validate_inputs(X, y, alternatives = None, varnames = varnames, asvars = asvars, isvars = isvars,
			base_alt = base_alt, fit_intercept = fit_intercept, max_iterations = max_iterations)

	with pytest.raises(ValueError): #without varnames
		model._validate_inputs(X, y, alternatives = [], varnames= None, asvars = asvars, isvars = isvars,
			base_alt = base_alt, fit_intercept = fit_intercept, max_iterations = max_iterations)
