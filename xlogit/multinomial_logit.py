"""Implements all the logic for multinomial and conditional logit models."""
# pylint: disable=line-too-long,invalid-name

import numpy as np
from ._choice_model import ChoiceModel
from ._optimize import _minimize, _numerical_hessian

"""
Notations
---------
    N : Number of choice situations
    J : Number of alternatives
    K : Number of variables
"""

_unpack_tuple = lambda x : x if len(x) > 1 else x[0]

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
            init_coeff=None, maxiter=2000, random_state=None, tol_opts=None, verbose=1,
            robust=False, num_hess=False, normalize_weights=False):
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

        normalize_weights: bool, default=False
            Whether weights should be normalized

        random_state : int, default=None
            Random seed for numpy random generator

        tol_opts : dict, default=None
            Options for tolerance of optimization routine. The dictionary accepts the following options (keys):

                ftol : float, default=1e-10
                    Tolerance for objective function (log-likelihood)

        verbose : int, default=1
            Verbosity of messages to show during estimation. 0: No messages,
            1: Some messages, 2: All messages


        Returns
        -------
        None.
        """
        X, y, varnames, alts, isvars, ids, weights, _, avail, _\
            = self._as_array(X, y, varnames, alts, isvars, ids, weights, None, avail, None)
        self._validate_inputs(X, y, alts, varnames, isvars, ids, weights)

        self._pre_fit(alts, varnames, isvars, base_alt, fit_intercept, maxiter)
        
        betas, X, y, weights, avail, Xnames = \
            self._setup_input_data(X, y, varnames, alts, ids, 
                                   isvars=isvars, weights=weights, avail=avail,
                                   init_coeff=init_coeff,
                                   random_state=random_state, verbose=verbose,
                                   predict_mode=False, normalize_weights=normalize_weights)

        tol = {'ftol': 1e-10}
        if tol_opts is not None:
            tol.update(tol_opts)

        # Call optimization routine
        fargs = (X, y, weights, avail)
        optim_res = _minimize(self._loglik_gradient, betas, args=fargs, method="BFGS", tol=tol['ftol'],
                              options={'maxiter': maxiter, 'disp': verbose > 1})  
        if num_hess:
            optim_res['hess_inv'] = _numerical_hessian(optim_res['x'], self._loglik_gradient, args=fargs)
        self._post_fit(optim_res, Xnames, X.shape[0], verbose, robust)


    def predict(self, X, varnames, alts, ids, isvars=None, weights=None,
                avail=None, random_state=None, verbose=1,
                return_proba=False, return_freq=False, normalize_weights=False):
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

        normalize_weights: bool, default=False
            Whether weights should be normalized

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
        X, _, varnames, alts, isvars, ids, weights, _, avail, _\
            = self._as_array(X, None, varnames, alts, isvars, ids, weights, None, avail, None)

        self._validate_inputs(X, None, alts, varnames, isvars, ids, weights)
       
        betas, X, _, weights, avail, Xnames = \
            self._setup_input_data(X, None, varnames, alts, ids, 
                                   isvars=isvars, weights=weights, avail=avail,
                                   init_coeff=self.coeff_,
                                   random_state=random_state, verbose=verbose,
                                   predict_mode=True, normalize_weights=normalize_weights)
        
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
            init_coeff=None, random_state=None, verbose=1, predict_mode=False,
            normalize_weights=False):
        X, y, _, avail = self._arrange_long_format(X, y, ids, alts, None, avail)
        y = self._format_choice_var(y, alts) if not predict_mode else None
        X, Xnames = self._setup_design_matrix(X)
        y = y.reshape(X.shape[0], X.shape[1])  if not predict_mode else None

        if random_state is not None:
            np.random.seed(random_state)

        if weights is not None and normalize_weights:
            weights = weights*(X.shape[0]/np.sum(weights))  # Normalize weights
                   
        if weights is not None:  # Reshape weights to match input data
            weights = weights[y.ravel().astype(bool)]

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

    def _loglik_gradient(self, betas, X, y, weights, avail, return_gradient=False):
        """Compute log-likelihood, gradient, and hessian."""
        p = self._compute_probabilities(betas, X, avail)
        # Log likelihood
        lik = np.sum(y*p, axis=1)
        loglik = np.log(lik)
        if weights is not None:
            loglik = loglik*weights
        loglik = np.sum(loglik)
        output = (-loglik, )
        # Individual contribution to the gradient
        if return_gradient:
            grad_n = np.einsum('nj,njk -> nk', (y-p), X)
            grad_n = grad_n if weights is None else grad_n*weights[:, None]
            grad = np.sum(grad_n, axis=0)
            output += (-grad.ravel(), )
            output += (grad_n, )
        return _unpack_tuple(output)

    def summary(self):
        """Show estimation results in console."""
        super(MultinomialLogit, self).summary()
