"""Implements all the logic for multinomial and conditional logit models."""
# pylint: disable=line-too-long,invalid-name

import numpy as np
from ._choice_model import ChoiceModel

"""
Notations
---------
    N : Number of choice situations
    J : Number of alternatives
    K : Number of variables
"""


class MultinomialLogit(ChoiceModel):
    """Class for estimation of Multinomial and Conditional Logit Models.

    Attributes
    ----------
        coeff_ : numpy array, shape (n_variables)
            Estimated coefficients

        coeff_names : numpy array, shape (n_variables)
            Names of the estimated coefficients

        stderr : numpy array, shape (n_variables)
            Standard errors of the estimated coefficients

        zvalues : numpy array, shape (n_variables)
            Z-values for t-distribution of the estimated coefficients

        pvalues : numpy array, shape (n_variables)
            P-values of the estimated coefficients

        loglikelihood : float
            Log-likelihood at the end of the estimation

        convergence : bool
            Whether convergence was reached during estimation

        total_iter : int
            Total number of iterations executed during estimation

        estim_time_sec : float
            Estimation time in seconds

        sample_size : int
            Number of samples used for estimation

        aic : float
            Akaike information criteria of the estimated model

        bic : float
            Bayesian information criteria of the estimated model
    """

    def fit(self, X, y, varnames=None, alts=None, isvars=None, ids=None,
            weights=None, avail=None, base_alt=None, fit_intercept=False,
            init_coeff=None, maxiter=2000, random_state=None, verbose=1):
        """Fit multinomial and/or conditional logit models.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_variables)
            Input data for explanatory variables in long format

        y : array-like, shape (n_samples,)
            Choices in long format

        varnames : list, shape (n_variables,)
            Names of explanatory variables that must match the number and
            order of columns in ``X``

        alts : array-like, shape (n_samples,)
            Alternative indexes in long format or list of alternative names

        isvars : list
            Names of individual-specific variables in ``varnames``

        ids : array-like, shape (n_samples,)
            Identifiers for choice situations in long format.

        weights : array-like, shape (n_variables,), default=None
            Weights for the choice situations in long format.

        avail: array-like, shape (n_samples,)
            Availability of alternatives for the choice situations. One when
            available or zero otherwise.

        base_alt : int, float or str, default=None
            Base alternative

        fit_intercept : bool, default=False
            Whether to include an intercept in the model.

        init_coeff : numpy array, shape (n_variables,), default=None
            Initial coefficients for estimation.

        maxiter : int, default=200
            Maximum number of iterations

        random_state : int, default=None
            Random seed for numpy random generator

        verbose : int, default=1
            Verbosity of messages to show during estimation. 0: No messages,
            1: Some messages, 2: All messages


        Returns
        -------
        None.
        """
        X, y, varnames, alts, isvars, ids, weights, _, avail\
            = self._as_array(X, y, varnames, alts, isvars, ids, weights, None,
                             avail)
        self._validate_inputs(X, y, alts, varnames, isvars, ids, weights,
                              base_alt, fit_intercept, maxiter)

        self._pre_fit(alts, varnames, isvars, base_alt, fit_intercept, maxiter)
        X, y, _ = self._arrange_long_format(X, y, ids, alts)

        if random_state is not None:
            np.random.seed(random_state)

        X, Xnames = self._setup_design_matrix(X)
        y = y.reshape(X.shape[0], X.shape[1])

        if avail is not None:
            avail = avail.reshape(X.shape[0], X.shape[1])

        if init_coeff is None:
            betas = np.repeat(.0, X.shape[2])
        else:
            betas = init_coeff
            if len(init_coeff) != X.shape[1]:
                raise ValueError("The size of initial_coeff must be: "
                                 + int(X.shape[1]))

        # Call optimization routine
        optimizat_res = self._bfgs_optimization(betas, X, y, weights, avail,
                                                maxiter)
        self._post_fit(optimizat_res, Xnames, X.shape[0], verbose)

    def _compute_probabilities(self, betas, X, avail):
        """Compute classic logit-based probabilities."""
        XB = X.dot(betas)
        eXB = np.exp(XB)
        if avail is not None:
            eXB = eXB*avail
        p = eXB/np.sum(eXB, axis=1, keepdims=True)  # (N,J)
        return p

    def _loglik_and_gradient(self, betas, X, y, weights, avail):
        """Compute log-likelihood, gradient, and hessian."""
        p = self._compute_probabilities(betas, X, avail)
        # Log likelihood
        lik = np.sum(y*p, axis=1)
        loglik = np.log(lik)
        if weights is not None:
            loglik = loglik*weights
        loglik = np.sum(loglik)
        # Individual contribution to the gradient
        grad = np.einsum('nj,njk -> nk', (y-p), X)
        if weights is not None:
            grad = grad*weights[:, None]

        H = np.dot(grad.T, grad)
        Hinv = np.linalg.inv(H)
        grad = np.sum(grad, axis=0)
        return -loglik, -grad, Hinv

    def summary(self):
        """Show estimation results in console."""
        super(MultinomialLogit, self).summary()

    def _bfgs_optimization(self, betas, X, y, weights, avail, maxiter):
        """BFGS optimization routine."""
        res, g, Hinv = self._loglik_and_gradient(betas, X, y, weights, avail)
        current_iteration = 0
        convergence = False
        while True:
            old_g = g

            d = -Hinv.dot(g)

            step = 2
            while True:
                step = step/2
                s = step*d
                betas = betas + s
                resnew, gnew, _ = self._loglik_and_gradient(betas, X, y,
                                                            weights, avail)
                if resnew <= res or step < 1e-10:
                    break

            old_res = res
            res = resnew
            g = gnew
            delta_g = g - old_g

            Hinv = Hinv + (((s.dot(delta_g) + (delta_g[None, :].dot(Hinv)).dot(
                delta_g))*np.outer(s, s)) / (s.dot(delta_g))**2) - ((np.outer(
                    Hinv.dot(delta_g), s) + (np.outer(s, delta_g)).dot(Hinv)) /
                    (s.dot(delta_g)))
            current_iteration = current_iteration + 1
            if np.abs(res - old_res) < 0.00001:
                convergence = True
                break
            if current_iteration > maxiter:
                convergence = False
                break

        return {'success': convergence, 'x': betas, 'fun': res,
                'hess_inv': Hinv, 'nit': current_iteration}
