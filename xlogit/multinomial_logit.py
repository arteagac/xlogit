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

    def fit(self, X, y, varnames, alts, ids, isvars=None,
            weights=None, avail=None, base_alt=None, fit_intercept=False,
            init_coeff=None, maxiter=2000, random_state=None, verbose=1):
        """Fit multinomial and/or conditional logit models.

        Parameters
        ----------
        X : array-like, shape (n_samples*n_alts, n_variables)
            Input data for explanatory variables in long format

        y : array-like, shape (n_samples*n_alts,)
            Chosen alternatives or one-hot encoded representation
            of the choices

        varnames : list, shape (n_variables,)
            Names of explanatory variables that must match the number and
            order of columns in ``X``

        alts : array-like, shape (n_samples*n_alts,)
            Alternative values in long format

        ids : array-like, shape (n_samples*n_alts,)
            Identifiers for the samples in long format.

        isvars : list, default=None
            Names of individual-specific variables in ``varnames``

        weights : array-like, shape (n_variables,), default=None
            Weights for the choice situations in long format.

        avail: array-like, shape (n_samples*n_alts,), default=None
            Availability of alternatives for the samples. One when
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
        self._validate_inputs(X, y, alts, varnames, isvars, ids, weights)

        self._pre_fit(alts, varnames, isvars, base_alt, fit_intercept, maxiter)
        
        betas, X, y, weights, avail, Xnames = \
            self._setup_input_data(X, y, varnames, alts, ids, 
                                   isvars=isvars, weights=weights, avail=avail,
                                   init_coeff=init_coeff,
                                   random_state=random_state, verbose=verbose,
                                   predict_mode=False)

        # Call optimization routine
        optimizat_res = self._bfgs_optimization(betas, X, y, weights, avail,
                                                maxiter)
        self._post_fit(optimizat_res, Xnames, X.shape[0], verbose)


    def predict(self, X, varnames, alts, ids, isvars=None, weights=None,
                avail=None, random_state=None, verbose=1,
                return_proba=False, return_freq=False):
        """Predict chosen alternatives.

        Parameters
        ----------
        X : array-like, shape (n_samples*n_alts, n_variables)
            Input data for explanatory variables in long format

        varnames : list, shape (n_variables,)
            Names of explanatory variables that must match the number and
            order of columns in ``X``

        alts : array-like, shape (n_samples*n_alts,)
            Alternative values in long format

        ids : array-like, shape (n_samples*n_alts,)
            Identifiers for the samples in long format.

        isvars : list
            Names of individual-specific variables in ``varnames``

        weights : array-like, shape (n_variables,), default=None
            Sample weights in long format.

        avail: array-like, shape (n_samples*n_alts,), default=None
            Availability of alternatives for the samples. One when
            available or zero otherwise.

        random_state : int, default=None
            Random seed for numpy random generator

        verbose : int, default=1
            Verbosity of messages to show during estimation. 0: No messages,
            1: Some messages, 2: All messages

        return_proba : bool, default=False
            If True, also return the choice probabilities

        return_freq : bool, default=False
            If True, also return the choice frequency for the alternatives
            

        Returns
        -------
        choices : array-like, shape (n_samples, )
            Chosen alternative for every sample in the dataset.

        proba : array-like, shape (n_samples, n_alts), optional
            Choice probabilities for each sample in the dataset. The 
            alternatives are ordered (in the columns) as they appear
            in ``self.alternatives``. Only provided if
            `return_proba` is True.

        freq : dict, optional
            Choice frequency for each alternative. Only provided
            if `return_freq` is True.
        """
        #=== 1. Preprocess inputs
        # Handle array-like inputs by converting everything to numpy arrays
        X, _, varnames, alts, isvars, ids, weights, _, avail = \
            self._as_array(X, None, varnames, alts, isvars, ids, weights, None,
                             avail)
        self._validate_inputs(X, None, alts, varnames, isvars, ids, weights)
       
        betas, X, _, weights, avail, Xnames = \
            self._setup_input_data(X, None, varnames, alts, ids, 
                                   isvars=isvars, weights=weights, avail=avail,
                                   init_coeff=self.coeff_,
                                   random_state=random_state, verbose=verbose,
                                   predict_mode=True)
        
        #=== 2. Compute choice probabilities
        proba = self._compute_probabilities(betas, X, avail)  # (N,J)
        
        #=== 3. Compute choices
        idx_max_proba = np.argmax(proba, axis=1)
        choices = self.alternatives[idx_max_proba]
        
        #=== 4. Arrange output depending on requested information
        output = (choices, )
        if return_proba:
            output += (proba, )
        
        if return_freq:
            alt_list, counts = np.unique(choices, return_counts=True)
            freq = dict(zip(list(alt_list),
                            list(np.round(counts/np.sum(counts), 3))))
            output += (freq, )
        
        _unpack_tuple = lambda x : x if len(x) > 1 else x[0]
        
        return _unpack_tuple(output) # Unpack before returning


    def _setup_input_data(self, X, y, varnames, alts, ids, isvars=None,
            weights=None, avail=None, base_alt=None, fit_intercept=False,
            init_coeff=None, random_state=None, verbose=1, predict_mode=False):
        X, y, _ = self._arrange_long_format(X, y, ids, alts)
        y = self._format_choice_var(y, alts) if not predict_mode else None
        X, Xnames = self._setup_design_matrix(X)
        y = y.reshape(X.shape[0], X.shape[1])  if not predict_mode else None

        if random_state is not None:
            np.random.seed(random_state)

        if weights is not None:
            weights = weights*(X.shape[0]/np.sum(weights))  # Normalize weights

        if avail is not None:
            avail = avail.reshape(X.shape[0], X.shape[1])

        if init_coeff is None:
            betas = np.repeat(.0, X.shape[2])
        else:
            betas = init_coeff
            if len(init_coeff) != X.shape[1]:
                raise ValueError("The size of initial_coeff must be: "
                                 + int(X.shape[1]))
        return betas, X, y, weights, avail, Xnames
        

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
