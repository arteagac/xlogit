# -*- coding: utf-8 -*-
import numpy as np
import pytest
from xlogit import MultinomialLogit

# Setup data used for tests
X = np.array([[2, 1], [1, 3], [3, 1], [2, 4], [2, 1], [2, 4]])
y = np.array([0, 1, 0, 1, 0, 1])
ids = np.array([1, 1, 2, 2, 3, 3])
alts = np.array([1, 2, 1, 2, 1, 2])
varnames = ["a", "b"]
N, J, K = 3, 2, 2


def test_log_likelihood():
    """
    Computes the log-likelihood "by hand" for a simple example and ensures
    that the one returned by xlogit is the same
    """
    X_, y_ = X.reshape(N, J, K), y.reshape(N, J)
    betas = np.array([.1, .1])

    # Compute log likelihood using xlogit
    model = MultinomialLogit()
    obtained_loglik, _, _ = model._loglik_and_gradient(betas, X_, y_, None,
                                                       None)

    # Compute expected log likelihood "by hand"
    eXB = np.exp(X_.dot(betas))
    expected_loglik = np.sum(np.log(
        np.sum(eXB/np.sum(eXB, axis=1, keepdims=True)*y_, axis=1)))

    assert pytest.approx(expected_loglik, obtained_loglik)

def test_predict():
    """
    Computes predictions "by hand" for a simple example and ensures that the
    probabilities returned by xlogit are the same
    """
    X_ = X.reshape(N, J, K)
    betas = np.array([.1, .1])

    #=== 1. Compute predictions using xlogit
    model = MultinomialLogit()
    model.alternatives =  np.array([1, 2])
    model.coeff_ = betas
    model._isvars, model._asvars, model._varnames = [], varnames, varnames
    model._fit_intercept = False
    model.coeff_names = np.array(["a", "b", "sd.a", "sd.b"])
    ypred, proba, freq = model.predict(X, varnames, alts, ids,
                                       return_proba=True,
                                       return_freq=True)
    
    #=== 2. Compute predictions by hand
    eXB = np.exp(X_.dot(betas))
    expec_proba = eXB/np.sum(eXB, axis=1, keepdims=True)
    expec_ypred = model.alternatives[np.argmax(expec_proba, axis=1)]
    alt_list, counts = np.unique(expec_ypred, return_counts=True)
    expec_freq = dict(zip(list(alt_list),
                          list(np.round(counts/np.sum(counts), 3))))
    #=== 3. Assert predictions are the same
    assert np.allclose(expec_proba, proba)
    assert np.array_equal(expec_ypred, ypred)
    assert expec_freq == freq
    


def test__bfgs_optimization():
    """
    Ensure that the bfgs optimization properly processes the input for one
    iteration. The value of 0.4044 was computed by hand for
    comparison purposes
    """
    X_, y_ = X.reshape(N, J, K), y.reshape(N, J)
    betas = np.array([.1, .1])
    model = MultinomialLogit()
    res = model._bfgs_optimization(betas, X_, y_, None, None, 0)

    assert pytest.approx(res['fun'], 0.40443136)


def test_fit():
    """
    Ensures the log-likelihood works for a single iterations with the default
    initial coefficients. The value of 0.4044 was computed by hand for
    comparison purposes
    """
    model = MultinomialLogit()
    model.fit(X, y, varnames=varnames, alts=alts, ids=ids,
              maxiter=0, verbose=0)

    assert pytest.approx(model.loglikelihood, -0.40443136)
