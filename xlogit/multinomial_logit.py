"""Implements all the logic for multinomial and conditional logit models."""
# pylint: disable=line-too-long,invalid-name

import numpy as np
from ._choice_model import ChoiceModel, diff_nonchosen_chosen
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
            init_coeff=None, maxiter=2000, random_state=None, tol_opts=None,
            verbose=1, robust=False, num_hess=False, scale_factor=None,
            addit=None, skip_std_errs=False):
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
            
        addit : array-like, shape (n_samples*n_alts, ), default=None
            Additive term to model coefficients kept fixed during estimation.
            
        base_alt : int, float or str, default=None
            Base alternative

        fit_intercept : bool, default=False
            Whether to include an intercept in the model.

        init_coeff : numpy array, shape (n_variables,), default=None
            Initial coefficients for estimation.

        maxiter : int, default=200
            Maximum number of iterations

        robust: bool, default=False
            Whether robust standard errors should be computed

        num_hess: bool, default=False
            Whether numerical hessian should be used for estimation of standard errors

        skip_std_errs: bool, default=False
            Whether estimation of standard errors should be skipped

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
        X, y, varnames, alts, isvars, ids, weights, _, avail, scale_factor, addit \
            = self._as_array(X, y, varnames, alts, isvars, ids, weights, None, avail, scale_factor, addit)
        self._validate_inputs(X, y, alts, varnames, isvars, ids, weights)

        self._pre_fit(alts, varnames, isvars, base_alt, fit_intercept, maxiter)
        
        betas, X, y, weights, avail, Xnames, scale, addit = \
            self._setup_input_data(X, y, varnames, alts, ids, 
                                   isvars=isvars, weights=weights, avail=avail,
                                   init_coeff=init_coeff,
                                   random_state=random_state, verbose=verbose,
                                   predict_mode=False, scale_factor=scale_factor, addit=addit)

        tol = {'ftol': 1e-10}
        if tol_opts is not None:
            tol.update(tol_opts)

        # Call optimization routine
        Xd, scale_d, addit_d, avail = diff_nonchosen_chosen(X, y, scale, addit, avail)  # Setup Xd as Xij - Xi*
        fargs = (Xd, scale_d, addit_d, weights, avail)
        optim_res = _minimize(self._loglik_gradient, betas, args=fargs, method="BFGS", tol=tol['ftol'],
                              options={'maxiter': maxiter, 'disp': verbose > 1})
        coef_names = Xnames
        if scale_factor is not None:
            coef_names = np.append(coef_names, "_scale_factor")

        if num_hess and not skip_std_errs:
            optim_res['hess_inv'] = _numerical_hessian(optim_res['x'], self._loglik_gradient, args=fargs)
        else:
            optim_res['hess_inv'] = np.eye(len(optim_res['x']))
        self._post_fit(optim_res, coef_names, X.shape[0], verbose, robust)


    def predict(self, X, varnames, alts, ids, isvars=None, weights=None,
                avail=None, random_state=None, verbose=1,
                return_proba=False, return_freq=False, scale_factor=None, addit=None):
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
            
        addit : array-like, shape (n_samples*n_alts, ), default=None
            Additive term to model coefficients kept fixed during estimation.
            
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
        X, _, varnames, alts, isvars, ids, weights, _, avail, scale_factor, addit \
            = self._as_array(X, None, varnames, alts, isvars, ids, weights, None, avail, scale_factor, addit)

        self._validate_inputs(X, None, alts, varnames, isvars, ids, weights)
       
        betas, X, _, weights, avail, Xnames, scale, addit = \
            self._setup_input_data(X, None, varnames, alts, ids, 
                                   isvars=isvars, weights=weights, avail=avail,
                                   init_coeff=self.coeff_,
                                   random_state=random_state, verbose=verbose,
                                   predict_mode=True, scale_factor=scale_factor, addit=addit)
            
        coeff_names = Xnames
        coeff_names = coeff_names if scale_factor is None else np.append(coeff_names, "_scale_factor")
        if not np.array_equal(coeff_names, self.coeff_names):
            raise ValueError("The provided 'varnames' yield coefficient names that are inconsistent with the stored "
                             "in 'self.coeff_names'")
        
        lambdac = 1 if scale_factor is None else betas[-1]
        sca = 0 if scale_factor is None else scale
        addit = 0 if addit is None else addit
        betas = betas if scale_factor is None else betas[:-1]

        #=== 2. Compute choice probabilities
        eV = np.exp(lambdac*(X.dot(betas) - sca + addit))
        eV = eV if avail is None else eV*avail
        proba = eV/np.sum(eV, axis=1, keepdims=True)  # (N,J)
        
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
            scale_factor=None, addit=None):
        self._check_long_format_consistency(ids, alts)
        y = self._format_choice_var(y, alts) if not predict_mode else None
        X, Xnames = self._setup_design_matrix(X)
        N, J, K = X.shape
        y = y.reshape(N, J)  if not predict_mode else None
        
        if random_state is not None:
            np.random.seed(random_state)
                   
        if weights is not None:  # Reshape weights to match input data
            weights = weights.reshape(N, J)[:, 0]

        if avail is not None:
            avail = avail.reshape(N, J)

        if init_coeff is None:
            betas = np.repeat(.0, K)
            betas = betas if scale_factor is None else np.append(betas, 1.)
        else:
            betas = init_coeff
            n_coeff = K + (0 if scale_factor is None else 1)
            if len(init_coeff) != n_coeff:
                raise ValueError(f"The size of initial_coeff must be: {n_coeff}")
        scale = None if scale_factor is None else scale_factor.reshape(N, J)
        addit = None if addit is None else addit.reshape(N, J)
        
        return betas, X, y, weights, avail, Xnames, scale, addit


    def _loglik_gradient(self, betas, Xd, scale_d, addit_d, weights, avail, return_gradient=True):
        """Compute log-likelihood, gradient, and hessian."""
        lambdac = 1 if scale_d is None else betas[-1]
        betas = betas if scale_d is None else betas[:-1]
        Xd = Xd if scale_d is None else lambdac*Xd
        scad = 0 if scale_d is None else lambdac*scale_d
        additd = 0 if addit_d is None else lambdac*addit_d
        #p = self._compute_probabilities(betas, X, avail)
        Vd = np.einsum('njk,k -> nj', Xd, betas) - scad + additd
        eVd = np.exp(Vd)
        eVd = eVd if avail is None else eVd*avail # Availablity of alts.
        proba = 1/(1+eVd.sum(axis=1))  # (N, )
        
        # Log likelihood
        lik = proba
        loglik = np.log(lik) if weights is None else np.log(lik)*weights
        loglik = np.sum(loglik)
        output = (-loglik, )
        # Individual contribution to the gradient
        if return_gradient:
            grad_n = -np.einsum('njk,nj -> nk', Xd, eVd)
            if scale_d is not None:
                gr_l = -np.einsum('nj,nj -> n', Vd/lambdac, eVd)[:, None]
                grad_n = np.append(grad_n, gr_l, 1)
            grad_n = grad_n*proba[:, None]
            grad_n = grad_n if weights is None else grad_n*weights[:, None]
            grad = np.sum(grad_n, axis=0)
            output += (-grad.ravel(), )
            output += (grad_n, )
        return _unpack_tuple(output)

    def summary(self):
        """Show estimation results in console."""
        super(MultinomialLogit, self).summary()
