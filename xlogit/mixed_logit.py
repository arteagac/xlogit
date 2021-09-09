"""Implements all the logic for mixed logit models."""

# pylint: disable=invalid-name
import scipy.stats
from scipy.optimize import minimize
from ._choice_model import ChoiceModel
from ._device import device as dev
import numpy as np

"""
Notations
---------
    N : Number of choice situations
    P : Number of observations per panel
    J : Number of alternatives
    K : Number of variables (Kf: fixed, Kr: random)
"""

MIN_COMP_ZERO = 1e-300
MAX_COMP_EXP = 700

class MixedLogit(ChoiceModel):
    """Class for estimation of Mixed Logit Models.

    Attributes
    ----------
        coeff_ : numpy array, shape (n_variables + n_randvars, )
            Estimated coefficients

        coeff_names : numpy array, shape (n_variables + n_randvars, )
            Names of the estimated coefficients

        stderr : numpy array, shape (n_variables + n_randvars, )
            Standard errors of the estimated coefficients

        zvalues : numpy array, shape (n_variables + n_randvars, )
            Z-values for t-distribution of the estimated coefficients

        pvalues : numpy array, shape (n_variables + n_randvars, )
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

    def __init__(self):
        """Init Function."""
        super(MixedLogit, self).__init__()
        self._rvidx = None  # Index of random variables (True when random var)
        self._rvdist = None  # List of mixing distributions of rand vars

    def fit(self, X, y, varnames, alts, ids, randvars, isvars=None, weights=None, avail=None,  panels=None,
            base_alt=None, fit_intercept=False, init_coeff=None, maxiter=2000, random_state=None, n_draws=1000,
            halton=True, verbose=1, batch_size=None, halton_opts=None, tol_opts=None):
        """Fit Mixed Logit models.

        Parameters
        ----------
        X : array-like, shape (n_samples*n_alts, n_variables)
            Input data for explanatory variables in long format

        y : array-like, shape (n_samples*n_alts,)
            Chosen alternatives or one-hot encoded representation of the choices

        varnames : list-like, shape (n_variables,)
            Names of explanatory variables that must match the number and order of columns in ``X``

        alts : array-like, shape (n_samples*n_alts,)
            Alternative values in long format

        ids : array-like, shape (n_samples*n_alts,)
            Identifiers for the samples in long format.

        randvars : dict
            Names (keys) and mixing distributions (values) of variables that have random parameters as coefficients.
            Possible mixing distributions are: ``'n'``: normal, ``'ln'``: lognormal, ``'u'``: uniform,
            ``'t'``: triangular, ``'tn'``: truncated normal

        isvars : list-like
            Names of individual-specific variables in ``varnames``

        weights : array-like, shape (n_samples,), default=None
            Sample weights in long format.

        avail: array-like, shape (n_samples*n_alts,), default=None
            Availability of alternatives for the choice situations. One when available or zero otherwise.

        panels : array-like, shape (n_samples*n_alts,), default=None
            Identifiers in long format to create panels in combination with ``ids``

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

        n_draws : int, default=500
            Number of random draws to approximate the mixing distributions of the random coefficients

        halton : bool, default=True
            Whether the estimation uses halton draws.
            
        halton_opts : dict, default=None
            Options for generation of halton draws. The dictionary accepts the following options (keys):

                shuffle : bool, default=False
                    Whether the Halton draws should be shuffled
                
                drop : int, default=100
                    Number of initial Halton draws to discard to minimize correlations between Halton sequences
                
                primes : list
                    List of primes to be used as base for generation of Halton sequences.

        tol_opts : dict, default=None
            Options for tolerance of optimization routine. The dictionary accepts the following options (keys):

                ftol : float
                    Tolerance for objective function (log-likelihood)
                
                gtol : float
                    Tolerance for gradient function.

        verbose : int, default=1
            Verbosity of messages to show during estimation. 0: No messages, 1: Some messages, 2: All messages

        batch_size : int, default=None
            Size of batches of random draws used to avoid overflowing memory during computations.

        Returns
        -------
        None.
        """
        # Handle array-like inputs by converting everything to numpy arrays
        X, y, varnames, alts, isvars, ids, weights, panels, avail\
            = self._as_array(X, y, varnames, alts, isvars, ids, weights,  panels, avail)

        self._validate_inputs(X, y, alts, varnames, isvars, ids, weights)
        
        self._pre_fit(alts, varnames, isvars, base_alt, fit_intercept, maxiter)
        
        batch_size = n_draws if batch_size is None else min(n_draws, batch_size)

        betas, X, y, panel_info, draws, weights, avail, Xnames = \
            self._setup_input_data(X, y, varnames, alts, ids, randvars, isvars=isvars, weights=weights, avail=avail,
                                   panels=panels, init_coeff=init_coeff, random_state=random_state, n_draws=n_draws,
                                   halton=halton, verbose=verbose, predict_mode=False, halton_opts=halton_opts)

        tol = {'ftol': 1e-5, 'gtol': 1e-4}
        if tol_opts is not None:
            tol.update(tol_opts)

        optimizat_res = \
            minimize(self._loglik_gradient, betas, jac=True, method='BFGS', tol=tol['ftol'],
                     args=(X, y, panel_info, draws, weights, avail, batch_size), 
                     options={'gtol': tol['gtol'], 'maxiter': maxiter, 'disp': verbose > 0})

        coef_names = np.append(Xnames, np.char.add("sd.", Xnames[self._rvidx]))

        self._post_fit(optimizat_res, coef_names, X.shape[0], verbose)


    def predict(self, X, varnames, alts, ids, isvars=None, weights=None, avail=None,  panels=None, random_state=None,
                n_draws=1000, halton=True, verbose=1, batch_size=None, return_proba=False, return_freq=False,
                halton_opts=None):
        """Predict chosen alternatives.

        Parameters
        ----------
        X : array-like, shape (n_samples*n_alts, n_variables)
            Input data for explanatory variables in long format

        varnames : list, shape (n_variables,)
            Names of explanatory variables that must match the number and order of columns in ``X``

        alts : array-like, shape (n_samples*n_alts,)
            Alternative values in long format

        ids : array-like, shape (n_samples*n_alts,)
            Identifiers for the samples in long format.

        isvars : list
            Names of individual-specific variables in ``varnames``

        weights : array-like, shape (n_variables,), default=None
            Sample weights in long format.

        avail: array-like, shape (n_samples*n_alts,), default=None
            Availability of alternatives for the samples. One when  available or zero otherwise.

        panels : array-like, shape (n_samples*n_alts,), default=None
            Identifiers in long format to create panels in combination with ``ids``

        random_state : int, default=None
            Random seed for numpy random generator

        n_draws : int, default=200
            Number of random draws to approximate the mixing distributions of the random coefficients

        halton : bool, default=True
            Whether the estimation uses halton draws.
            
        halton_opts : dict, default=None
            Options for generation of Halton draws. The dictionary accepts the following options (keys):
            
                shuffle : bool, default=False
                    Whether the Halton draws should be shuffled
                
                drop : int, default=100
                    Number of initial Halton draws to discard to minimize correlations between Halton sequences
                
                primes : list
                    List of primes to be used as base for generation of Halton sequences.

        verbose : int, default=1
            Verbosity of messages to show during estimation. 0: No messages, 1: Some messages, 2: All messages

        batch_size : int, default=None
            Size of batches of random draws used to avoid overflowing memory during computations.
            
        return_proba : bool, default=False
            If True, also return the choice probabilities

        return_freq : bool, default=False
            If True, also return the frequency of the chosen the alternatives


        Returns
        -------
        choices : array-like, shape (n_samples, )
            Chosen alternative for every sample in the dataset.

        proba : array-like, shape (n_samples, n_alts), optional
            Choice probabilities for each sample in the dataset. The alternatives are ordered (in the columns) as they 
            appear in ``self.alternatives``. Only provided if `return_proba` is True.

        freq : dict, optional
            Choice frequency for each alternative. Only provided if `return_freq` is True.
        """
        # Handle array-like inputs by converting everything to numpy arrays
        #=== 1. Preprocess inputs
        X, _, varnames, alts, isvars, ids, weights, panels, avail\
            = self._as_array(X, None, varnames, alts, isvars, ids, weights, panels, avail)
        
        self._validate_inputs(X, None, alts, varnames, isvars, ids, weights)
        
        betas, X, _, panel_info, draws, weights, avail, Xnames = \
            self._setup_input_data(X, None, varnames, alts, ids, self.randvars,  isvars=isvars, weights=weights,
                                   avail=avail, panels=panels, init_coeff=self.coeff_, random_state=random_state,
                                   n_draws=n_draws, halton=halton, verbose=verbose, predict_mode=True,
                                   halton_opts=halton_opts)
        
        coef_names = np.append(Xnames, np.char.add("sd.", Xnames[self._rvidx]))
        if not np.array_equal(coef_names, self.coeff_names):
            raise ValueError("The provided 'varnames' yield coefficient names that are inconsistent with the stored "
                             "in 'self.coeff_names'")
        
        betas = dev.to_gpu(betas) if dev.using_gpu else betas
        
        #=== 2. Compute choice probabilities
        batch_size = n_draws if batch_size is None else min(n_draws, batch_size)
        R = draws.shape[-1]
        
        p = []        
        n_batches = R//batch_size + (1 if R % batch_size != 0 else 0)
        for batch in range(n_batches):
            draws_batch = draws[:, :, batch*batch_size: batch*batch_size + batch_size]
            if dev.using_gpu:
                draws_batch = dev.to_gpu(draws_batch)

            p_batch = self._compute_probabilities(betas, X, panel_info, draws_batch, avail)  # (N,P,J,R)
            p_batch = self._prob_product_across_panels(p_batch, panel_info)  # (N,J,R)
            
            if dev.using_gpu:
                p_batch = dev.to_cpu(p_batch)
            p.append(p_batch)
            draws_batch = None
        p = np.concatenate(p, axis=-1) 
        proba = p.mean(axis=-1)   # (N,J)
        
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
 
    def _setup_input_data(self, X, y, varnames, alts, ids, randvars, isvars=None, weights=None, avail=None,
                          panels=None, init_coeff=None, random_state=None, n_draws=200, halton=True, verbose=1,
                          predict_mode=False, halton_opts=None):
        if random_state is not None:
            np.random.seed(random_state)

        X, y, panels = self._arrange_long_format(X, y, ids, alts, panels)
        y = self._format_choice_var(y, alts) if not predict_mode else None
        X, Xnames = self._setup_design_matrix(X)
        self._model_specific_validations(randvars, Xnames)

        J, K, R = X.shape[1], X.shape[2], n_draws
        Kr = len(randvars)

        if panels is not None:  # If panel
            X, y, panel_info = self._balance_panels(X, y, panels)
            N, P = panel_info.shape
        else:
            N, P = X.shape[0], 1
            panel_info = np.ones((N, 1))

        # Reshape arrays in the format required for the rest of the estimation
        X = X.reshape(N, P, J, K)
        y = y.reshape(N, P, J, 1) if not predict_mode else None

        if not predict_mode:
            self._setup_randvars_info(randvars, Xnames)
        self.n_draws = n_draws
        self.verbose = verbose

        if weights is not None:
            weights = weights*(N/np.sum(weights))  # Normalize weights

        if avail is not None:
            avail = avail.reshape(N, P, J)

        # Generate draws
        draws = self._generate_draws(N, R, halton, halton_opts=halton_opts)  # (N,Kr,R)
        if init_coeff is None:
            betas = np.repeat(.1, K + Kr)
        else:
            betas = init_coeff
            if len(init_coeff) != K + Kr:
                raise ValueError("The size of init_coeff must be: " + K + Kr)

        # Move data to GPU if GPU is being used
        if dev.using_gpu:
            X = dev.to_gpu(X)
            y = dev.to_gpu(y) if not predict_mode else None
            panel_info = dev.to_gpu(panel_info)
            #draws = dev.to_gpu(draws)
            if weights is not None:
                weights = dev.to_gpu(weights)
            if avail is not None:
                avail = dev.to_gpu(avail)
            if verbose > 0:
                print("GPU processing enabled.")
        return betas, X, y, panel_info, draws, weights, avail, Xnames


    def _setup_randvars_info(self, randvars, Xnames):
        self.randvars = randvars
        self._rvidx, self._rvdist = [], []
        for var in Xnames:
            if var in self.randvars.keys():
                self._rvidx.append(True)
                self._rvdist.append(self.randvars[var])
            else:
                self._rvidx.append(False)
        self._rvidx = np.array(self._rvidx)


    def _compute_probabilities(self, betas, X, panel_info, draws, avail):
        """Compute the standard logit-based probabilities.

        Random and fixed coefficients are handled separately.
        """
        Bf, Br = self._transform_betas(betas, draws)  # Get fixed and rand coef
        Xf = X[:, :, :, ~self._rvidx]  # Data for fixed coefficients
        Xr = X[:, :, :, self._rvidx]   # Data for random coefficients

        XBf = dev.cust_einsum('npjk,k -> npj', Xf, Bf)  # (N,P,J)
        XBr = dev.cust_einsum('npjk,nkr -> npjr', Xr, Br)  # (N,P,J,R)
        V = XBf[:, :, :, None] + XBr  # (N,P,J,R)
        
        V[V > MAX_COMP_EXP] = MAX_COMP_EXP
        eV = dev.np.exp(V)

        if avail is not None:
            eV = eV*avail[:, :, :, None]  # Acommodate availablity of alts.

        sumeV = dev.np.sum(eV, axis=2, keepdims=True)
        sumeV[sumeV == 0] = MIN_COMP_ZERO # 
        p = eV/sumeV  # (N,P,J,R)
        p = p*panel_info[:, :, None, None]  # Zero for unbalanced panels
        return p  # (N,P,J,R)

    def _loglik_gradient(self, betas, X, y, panel_info, draws, weights, avail, batch_size):
        """Compute the log-likelihood and gradient.

        Fixed and random parameters are handled separately to speed up the estimation and the results are concatenated.
        """
        if dev.using_gpu:
            betas = dev.to_gpu(betas)
        
        N, R, Kf, Kr = X.shape[0], draws.shape[-1], np.sum(~self._rvidx), np.sum(self._rvidx)
        
        gr_f, gr_b, gr_w, pch = np.zeros((N, Kf)), np.zeros((N, Kr)), np.zeros((N, Kr)), []  # Batch data

        n_batches = R//batch_size + (1 if R % batch_size != 0 else 0)
        for batch in range(n_batches):
            draws_batch = draws[:, :, batch*batch_size: batch*batch_size + batch_size]
            if dev.using_gpu:
                draws_batch = dev.to_gpu(draws_batch)
            
            p = self._compute_probabilities(betas, X, panel_info, draws_batch, avail)

            # Probability of chosen alternatives
            pch_batch = (y*p).sum(axis=2)  # (N,P,R)
            pch_batch = self._prob_product_across_panels(pch_batch, panel_info)  # (N,R)

            # Gradient
            Xf = X[:, :, :, ~self._rvidx]
            Xr = X[:, :, :, self._rvidx]
    
            ymp = y - p  # (N,P,J,R)
            # Gradient for fixed and random params
            gr_f_batch = dev.cust_einsum('npjr,npjk -> nkr', ymp, Xf)
            der = self._compute_derivatives(betas, draws_batch)
            gr_b_batch = dev.cust_einsum('npjr,npjk -> nkr', ymp, Xr)*der
            gr_w_batch = dev.cust_einsum('npjr,npjk -> nkr', ymp, Xr)*der*draws_batch

            gr_f_batch = (gr_f_batch*pch_batch[:, None, :]).sum(axis=-1)  # (N,K,R)*(N,1,R)
            gr_b_batch = (gr_b_batch*pch_batch[:, None, :]).sum(axis=-1)
            gr_w_batch = (gr_w_batch*pch_batch[:, None, :]).sum(axis=-1)

            
            #Save batch results
            if dev.using_gpu:
                gr_f_batch = dev.to_cpu(gr_f_batch)
                gr_b_batch, gr_w_batch = dev.to_cpu(gr_b_batch), dev.to_cpu(gr_w_batch)
                pch_batch = dev.to_cpu(pch_batch)

            # Accumulate to later compute mean
            gr_f += gr_f_batch
            gr_b += gr_b_batch
            gr_w += gr_w_batch

            pch.append(pch_batch)
            draws_batch, p, der, ymp = None, None, None, None  # Release GPU memory
            
        pch = np.concatenate(pch, axis=-1) 
        # Log-likelihood
        lik = pch.mean(axis=1)  # (N,)
        
        gr_f = (gr_f/R)/lik[:, None]
        gr_b = (gr_b/R)/lik[:, None]
        gr_w = (gr_w/R)/lik[:, None]

        # Put all gradients in a single array and aggregate them
        grad = self._concat_gradients(gr_f, gr_b, gr_w)  # (N,K)
        if weights is not None:
            grad = grad*weights[:, None]
        grad = grad.sum(axis=0)  # (K,)

        # Log-likelihood
        loglik = np.log(lik)
        if weights is not None:
            loglik = loglik*weights
        loglik = loglik.sum()
        
        self.total_fun_eval += 1
        if self.verbose > 1:
            print(f"Evaluation {self.total_fun_eval} Log-Lik.={-loglik:.2f}")
        return -loglik.ravel(), -grad.ravel()

    def _concat_gradients(self, gr_f, gr_b, gr_w):
        idx = np.append(np.where(~self._rvidx)[0], np.where(self._rvidx)[0])
        gr_fb = np.concatenate((gr_f, gr_b), axis=1)[:, idx]
        return np.concatenate((gr_fb, gr_w), axis=1)

    def _prob_product_across_panels(self, prob, panel_info):
        if not np.all(panel_info):  # If panel unbalanced. Not all ones
            prob[panel_info==0, :] = 1  # Multiply by one when unbalanced
        prob = prob.prod(axis=1)  # (N,R)
        #MIN_COMP_ZERO = np.finfo(prob.dtype).max*.9  # ~1e-300 for float64
        prob[prob == 0] = MIN_COMP_ZERO
        return prob  # (N,R)

    def _apply_distribution(self, betas_random):
        """Apply the mixing distribution to the random betas."""
        for k, dist in enumerate(self._rvdist):
            if dist == 'ln':
                betas_random[:, k, :] = dev.np.exp(betas_random[:, k, :])
            elif dist == 'tn':
                betas_random[:, k, :] = betas_random[:, k, :] *\
                    (betas_random[:, k, :] > 0)
        return betas_random

    def _balance_panels(self, X, y, panels):
        """Balance panels if necessary and produce a new version of X and y.

        If panels are already balanced, the same X and y are returned. This also returns panel_info, which keeps track
        of the panels that needed balancing.
        """
        _, J, K = X.shape
        _, p_obs = np.unique(panels, return_counts=True)
        p_obs = (p_obs/J).astype(int)
        N = len(p_obs)  # This is the new N after accounting for panels
        P = np.max(p_obs)  # Panel length for all records

        if not np.all(p_obs[0] == p_obs):  # Balancing needed
            y = y.reshape(X.shape[0], J, 1) if y is not None else None
            Xbal, ybal = np.zeros((N*P, J, K)), np.zeros((N*P, J, 1))
            panel_info = np.zeros((N, P))
            cum_p = 0  # Cumulative sum of n_obs at every iteration
            for n, p in enumerate(p_obs):
                # Copy data from original to balanced version
                Xbal[n*P:n*P + p, :, :] = X[cum_p:cum_p + p, :, :]
                ybal[n*P:n*P + p, :, :] = y[cum_p:cum_p + p, :, :] if y is not None else None  # if in predict mode
                panel_info[n, :p] = np.ones(p)
                cum_p += p

        else:  # No balancing needed
            Xbal, ybal = X, y
            panel_info = np.ones((N, P))
        ybal = ybal if y is not None else None  # in predict mode
        return Xbal, ybal, panel_info

    def _compute_derivatives(self, betas, draws):
        """Compute the derivatives based on the mixing distributions."""
        N, R, Kr = draws.shape[0], draws.shape[2], self._rvidx.sum()
        der = dev.np.ones((N, Kr, R), dtype=draws.dtype)
        if any(set(self._rvdist).intersection(['ln', 'tn'])):
            _, betas_random = self._transform_betas(betas, draws)
            for k, dist in enumerate(self._rvdist):
                if dist == 'ln':
                    der[:, k, :] = betas_random[:, k, :]
                elif dist == 'tn':
                    der[:, k, :] = 1*(betas_random[:, k, :] > 0)
        return der

    def _transform_betas(self, betas, draws):
        """Compute the products between the betas and the random coefficients.

        This method also applies the associated mixing distributions
        """
        # Extract coeffiecients from betas array
        betas_fixed = betas[np.where(~self._rvidx)[0]]
        br_mean = betas[np.where(self._rvidx)[0]]
        br_sd = betas[len(self._rvidx):]  # Last Kr positions
        # Compute: betas = mean + sd*draws
        betas_random = br_mean[None, :, None] + draws*br_sd[None, :, None]
        betas_random = self._apply_distribution(betas_random)
        return betas_fixed, betas_random

    def _generate_draws(self, sample_size, n_draws, halton=True, halton_opts=None):
        """Generate draws based on the given mixing distributions."""
        if halton:
            draws = self._generate_halton_draws(sample_size, n_draws, len(self._rvdist),
                                                **halton_opts if halton_opts is not None else {})
        else:
            draws = self._generate_random_draws(sample_size, n_draws, len(self._rvdist))

        for k, dist in enumerate(self._rvdist):
            if dist in ['n', 'ln', 'tn']:  # Normal based
                draws[:, k, :] = scipy.stats.norm.ppf(draws[:, k, :])
            elif dist == 't':  # Triangular
                draws_k = draws[:, k, :]
                draws[:, k, :] = (np.sqrt(2*draws_k) - 1)*(draws_k <= .5) +\
                    (1 - np.sqrt(2*(1 - draws_k)))*(draws_k > .5)
            elif dist == 'u':  # Uniform
                draws[:, k, :] = 2*draws[:, k, :] - 1

        return draws  # (N,Kr,R)

    def _generate_random_draws(self, sample_size, n_draws, n_vars):
        """Generate random uniform draws between 0 and 1."""
        return np.random.uniform(size=(sample_size, n_vars, n_draws))

    def _generate_halton_draws(self, sample_size, n_draws, n_vars, shuffled=False, drop=100, primes=None):
        """Generate Halton draws for multiple random variables using different primes as base"""
        if primes is None:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 71, 73, 79, 83, 89, 97, 101,
                      103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
                      199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311]
        
        def halton_seq(length, prime=3, shuffled=False, drop=100):
            """Generates a halton sequence while handling memory efficiently.
            
            Memory is efficiently handled by creating a single array ``seq`` that is iteratively filled without using
            intermidiate arrays.
            """
            req_length = length + drop
            seq = np.empty(req_length)
            seq[0] = 0
            seq_idx = 1
            t=1
            while seq_idx < req_length:
                d = 1/prime**t
                seq_size = seq_idx
                i = 1
                while i < prime and seq_idx < req_length:
                    max_seq = min(req_length - seq_idx, seq_size)
                    seq[seq_idx: seq_idx+max_seq] = seq[:max_seq] + d*i
                    seq_idx += max_seq
                    i += 1
                t += 1
            seq = seq[drop:length+drop]
            if shuffled:
                np.random.shuffle(seq)
            return seq

        draws = [halton_seq(sample_size*n_draws, prime=primes[i % len(primes)],
                            shuffled=shuffled, drop=drop).reshape(sample_size, n_draws) for i in range(n_vars)]
        draws = np.stack(draws, axis=1)
        return draws  # (N,Kr,R)

    def _model_specific_validations(self, randvars, Xnames):
        """Conduct validations specific for mixed logit models."""
        if randvars is None:
            raise ValueError("The 'randvars' parameter is required for Mixed Logit estimation")
        if not set(randvars.keys()).issubset(Xnames):
            raise ValueError("Some variable names in 'randvars' were not found in the list of variable names")
        if not set(randvars.values()).issubset(["n", "ln", "t", "tn", "u"]):
            raise ValueError("Wrong mixing distribution in 'randvars'. Accepted distrubtions are n, ln, t, u, tn")

    def summary(self):
        """Show estimation results in console."""
        super(MixedLogit, self).summary()

    @staticmethod
    def check_if_gpu_available():
        """Check if GPU processing is available by running a quick estimation.

        Returns
        -------
        bool
            True if GPU processing is available, False otherwise.

        """
        n_gpus = dev.get_device_count()
        if n_gpus > 0:
            # Test a very simple example to see if CuPy is working
            X = np.array([[2, 1], [1, 3], [3, 1], [2, 4]])
            y = np.array([0, 1, 0, 1])
            alts = np.array([1, 2, 1, 2])
            ids = np.array([1, 2, 3, 4])
            model = MixedLogit()
            model.fit(X, y, varnames=["a", "b"], ids=ids, alts=alts,
                      randvars={'a': 'n', 'b': 'n'}, maxiter=0, n_draws=500,
                      verbose=0)
            print("{} GPU device(s) available. xlogit will use "
                  "GPU processing".format(n_gpus))
            return True
        else:
            print("*** No GPU device found. Verify CuPy is properly installed")
            return False
