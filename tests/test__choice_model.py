# -*- coding: utf-8 -*-
import numpy as np
import pytest
from xlogit import MultinomialLogit

X = np.array([[2, 1], [1, 3], [3, 1], [2, 4], [2, 1], [2, 4]])
y = np.array([0, 1, 0, 1, 0, 1])
ids = np.array([1, 1, 2, 2, 3, 3])
alts = np.array([1, 2, 1, 2, 1, 2])
varnames = ["a", "b"]
N, J, K = 3, 2, 2


def test__setup_design_matrix():
    """
    Ensures that xlogit properly adds an intercept when necessary

    """
    model = MultinomialLogit()
    model._pre_fit(alts, varnames, isvars=None, base_alt=None,
                   fit_intercept=True, maxiter=0)
    X_, Xnames_ = model._setup_design_matrix(X)
    assert X_.shape == (3, 2, 3)
    assert list(Xnames_) == ["_intercept.2", "a", "b"]


def test__validate_inputs():
    """
    Covers potential mistakes in parameters of the fit method that xlogit
    should be able to identify
    """
    model = MultinomialLogit()
    validate = model._validate_inputs
    with pytest.raises(ValueError):  # match between columns in X and varnames
        validate(X, y, alts, varnames=["a"], isvars=None, ids=ids,
                 weights=None, base_alt=None, fit_intercept=True, maxiter=0)

    with pytest.raises(ValueError):  # alts can't be None
        validate(X, y, None, varnames=varnames, isvars=None, ids=ids,
                 weights=None, base_alt=None, fit_intercept=True, maxiter=0)

    with pytest.raises(ValueError):  # varnames can't be None
        validate(X, y, alts, varnames=None, isvars=None, ids=ids,
                 weights=None, base_alt=None, fit_intercept=True, maxiter=0)

    with pytest.raises(ValueError):  # X dimensions
        validate(np.array([]), y, alts, varnames=None, isvars=None, ids=ids,
                 weights=None, base_alt=None, fit_intercept=True, maxiter=0)

    with pytest.raises(ValueError):  # y dimensions
        validate(X, np.array([]), alts, varnames=None, isvars=None, ids=ids,
                 weights=None, base_alt=None, fit_intercept=True, maxiter=0)

def test__format_choice_var():
    """
    Ensures that the variable y is properly formatted as needed by internal 
    procedures regardless of the input data type.
    """
    model = MultinomialLogit()
    expected = np.array([1, 0, 0, 1, 1, 0])
    
    y1 = np.array([1, 1, 2, 2, 1, 1])
    assert np.array_equal(model._format_choice_var(y1, alts), expected)
    
    y2 = np.array(['a', 'a', 'b', 'b', 'a', 'a'])
    alts2 = np.array(['a', 'b', 'a', 'b', 'a', 'b',])
    assert np.array_equal(model._format_choice_var(y2, alts2), expected)
    

def test_summary():
    """
    Ensures that calls to summary when the model has not been fit warns the
    user
    """
    model = MultinomialLogit()
    with pytest.warns(UserWarning):
        model.summary()
