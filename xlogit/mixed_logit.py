"""Implements all the logic for mixed logit models."""

# pylint: disable=invalid-name
import scipy.stats
from scipy.optimize import minimize, approx_fprime
from ._choice_model import ChoiceModel
from ._device import device as dev
from .multinomial_logit import MultinomialLogit
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

_unpack_tuple = lambda x : x if len(x) > 1 else x[0]

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
            halton=True, verbose=1, batch_size=None, halton_opts=None, tol_opts=None, robust=False, num_hess=False):
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

                ftol : float, default=1e-10
                    Tolerance for objective function (log-likelihood)
                
                gtol : float, default=1e-5
                    Tolerance for gradient function.

        verbose : int, default=1
            Verbosity of messages to show during estimation. 0: No messages, 1: Some messages, 2: All messages

        batch_size : int or dict, default=None
            Size of batches used to avoid memory overflow. If `int`, the value is the batch size across draws.
            If dict, the following options control batching across samples and draws:

                samples : int, default=n_samples
                    Batch size across samples

                draws : int, default=n_draws
                    Batch size across draws

        Returns
        -------
        None.
        """
        # Handle array-like inputs by converting everything to numpy arrays
        X, y, varnames, alts, isvars, ids, weights, panels, avail\
            = self._as_array(X, y, varnames, alts, isvars, ids, weights,  panels, avail)

        self._validate_inputs(X, y, alts, varnames, isvars, ids, weights)

        if init_coeff is None:
            # Initialize coefficients using a multinomial logit model
            mnl = MultinomialLogit()
            mnl.fit(X, y, varnames, alts, ids, isvars=isvars, weights=weights,
                    avail=avail, base_alt=base_alt, fit_intercept=fit_intercept)
            init_coeff = np.concatenate((mnl.coeff_, np.repeat(.1, len(randvars))))

        self._pre_fit(alts, varnames, isvars, base_alt, fit_intercept, maxiter)

        betas, X, y, panels, draws, weights, avail, Xnames = \
            self._setup_input_data(X, y, varnames, alts, ids, randvars, isvars=isvars, weights=weights, avail=avail,
                                   panels=panels, init_coeff=init_coeff, random_state=random_state, n_draws=n_draws,
                                   halton=halton, verbose=verbose, predict_mode=False, halton_opts=halton_opts)

        tol = {'ftol': 1e-10, 'gtol': 1e-4}
        if tol_opts is not None:
            tol.update(tol_opts)

        batch_sizes = self._setup_batch_sizes(batch_size, X, draws)

        optimizat_res = self._bfgs_optimization(betas, X, y, panels, draws, weights, avail, batch_sizes, maxiter, tol['ftol'], num_hess=num_hess)

        coef_names = np.append(Xnames, np.char.add("sd.", Xnames[self._rvidx]))

        self._post_fit(optimizat_res, coef_names, X.shape[0], verbose, robust)


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

        batch_size : int or dict, default=None
            Size of batches used to avoid memory overflow. If `int`, the value is the batch size across draws.
            If dict, the following options control batching across samples and draws:

                samples : int, default=n_samples
                    Batch size across samples

                draws : int, default=n_draws
                    Batch size across draws

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
        
        betas, X, _, panels, draws, weights, avail, Xnames = \
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
        batch_sizes = self._setup_batch_sizes(batch_size, X, draws)
        R = draws.shape[-1]
        
        p = []       
        batch_size = batch_sizes['draws'] 
        n_batches = R//batch_size + (1 if R % batch_size != 0 else 0)
        for batch in range(n_batches):
            draws_batch = draws[:, :, batch*batch_size: batch*batch_size + batch_size]
            draws_batch = dev.to_gpu(draws_batch)

            p_batch = self._batch_s_compute_probabilities(betas, X, draws_batch, avail, batch_sizes)  # (N,J,R)
            p_batch = self._prob_product_across_panels(p_batch, panels)  # (Np,J,R)

            p.append(dev.to_cpu(p_batch))
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
      
        return _unpack_tuple(output) # Unpack before returning
 
    def _setup_input_data(self, X, y, varnames, alts, ids, randvars, isvars=None, weights=None, avail=None,
                          panels=None, init_coeff=None, random_state=None, n_draws=200, halton=True, verbose=1,
                          predict_mode=False, halton_opts=None):
        if random_state is not None:
            np.random.seed(random_state)

        X, y, panels, avail = self._arrange_long_format(X, y, ids, alts, panels, avail)
        y = self._format_choice_var(y, alts) if not predict_mode else None
        X, Xnames = self._setup_design_matrix(X)
        self._model_specific_validations(randvars, Xnames)

        N, J, K, R = X.shape[0], X.shape[1], X.shape[2], n_draws
        Kr = len(randvars)

        if panels is not None:
            # Convert panel ids to indexes 
            panels = panels.reshape(N, J)[:, 0]
            panels_idx = np.empty(N)
            for i, u in enumerate(np.unique(panels)):
                panels_idx[np.where(panels==u)] = i
            panels = panels_idx.astype(int)

        # Reshape arrays in the format required for the rest of the estimation
        X = X.reshape(N, J, K)
        y = y.reshape(N, J, 1) if not predict_mode else None

        if not predict_mode:
            self._setup_randvars_info(randvars, Xnames)
        self.n_draws = n_draws
        self.verbose = verbose

        if weights is not None:
            weights = weights*(N/np.sum(weights))  # Normalize weights

        if avail is not None:
            avail = avail.reshape(N, J)

        # Generate draws
        n_samples = N if panels is None else panels[-1] + 1
        draws = self._generate_draws(n_samples, R, halton, halton_opts=halton_opts)
        draws = draws if panels is None else draws[panels]  # (N,Kr,R)

        if init_coeff is None:
            betas = np.repeat(.1, K + Kr)
        else:
            betas = init_coeff
            if len(init_coeff) != K + Kr:
                raise ValueError("The size of init_coeff must be: {}".format(K + Kr))

        if dev.using_gpu and verbose > 0:
            print("GPU processing enabled.")
        return betas, X, y, panels, draws, weights, avail, Xnames

    def _setup_batch_sizes(self, batch_size, X, draws):
        batch_sizes = {'samples': X.shape[0], 'draws': draws.shape[-1]}
        if batch_size is not None:
            if isinstance(batch_size, int):
                batch_sizes['draws'] = batch_size
            else:
                batch_sizes.update(batch_size)
        return batch_sizes


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
        Xf = X[:, :, ~self._rvidx]  # Data for fixed coefficients
        Xr = X[:, :, self._rvidx]   # Data for random coefficients

        XBf = dev.cust_einsum('njk,k -> nj', Xf, Bf)  # (N,J)
        XBr = dev.cust_einsum('njk,nkr -> njr', Xr, Br)  # (N,J,R)
        V = XBf[:, :, None] + XBr  # (N,J,R)

        eV = dev.np.exp(V)
        eV = eV if avail is None else eV*avail[:, :, None]  # Availablity of alts.
        sumeV = dev.np.sum(eV, axis=1, keepdims=True)  # (N,1,R)
        p = eV/sumeV  # (N,J,R)
        return p  # (N,J,R)

    def _batch_s_compute_probabilities(self, betas, X, draws, avail, batch_sizes):
        """Wrapper to batch probability computation across samples"""
        batch_size = batch_sizes['samples']
        N = X.shape[0]
        n_batches = N//batch_size + (1 if N % batch_size != 0 else 0)
        p = []
        for batch in range(n_batches):
            s_start, s_end = batch*batch_size, batch*batch_size + batch_size
            X_batch, avail_batch = X[s_start: s_end], avail[s_start:s_end] if avail is not None else None
            if dev.using_gpu:
                X_batch, avail_batch = dev.to_gpu(X_batch),  dev.to_gpu(avail_batch)
            p_batch = self._compute_probabilities(betas, X_batch, None, draws[s_start:s_end], avail_batch)# (N_,J,R_)
            X_batch, avail_batch = None, None

            p_batch = dev.to_cpu(p_batch) if dev.using_gpu else p_batch
            p.append(p_batch)
        p = np.concatenate(p, axis=0)  # (N,J,R_)
        return p


    def _loglik_gradient(self, betas, X, y, panels, draws, weights, avail, batch_sizes, return_gradient=False):
        """Compute the log-likelihood and gradient.

        Fixed and random parameters are handled separately to speed up the estimation and the results are concatenated.
        """
        betas = dev.to_gpu(betas)
        batch_size = batch_sizes['draws']
        N, J, R, Kf, Kr = X.shape[0], X.shape[1], draws.shape[-1], np.sum(~self._rvidx), np.sum(self._rvidx)
        
        gr_f, gr_b, gr_w, pch = np.zeros((N, Kf)), np.zeros((N, Kr)), np.zeros((N, Kr)), []  # Batch data
        pcp_f, pcp_b, pcp_w, ch_b, ch_w = np.zeros((N, J)), np.zeros((N, J, Kr)), np.zeros((N, J, Kr)), np.zeros((N, Kr)), np.zeros((N, Kr))
        n_batches = R//batch_size + (1 if R % batch_size != 0 else 0)
        #if not compute_grad:
        for batch in range(n_batches):
            draws_batch = draws[:, :, batch*batch_size: batch*batch_size + batch_size]
            draws_batch = dev.to_gpu(draws_batch)
            
            prob = self._batch_s_compute_probabilities(betas, X, draws_batch, avail, batch_sizes)  # (N,J,R)

            # Probability of chosen alternatives
            pch_batch = dev.nan_safe_sum(y*prob, axis=1)  # (N,R)
            pch_batch = self._prob_product_across_panels(pch_batch, panels) # (Np,R)
            pch.append(dev.to_cpu(pch_batch))

            # Gradient
            if return_gradient:
                pch_batch = pch_batch if panels is None else pch_batch[panels]   # (N,R)
                pcp_batch = prob*pch_batch[:, None, :]  # (N,J,R)

                der = self._compute_derivatives(betas, draws_batch)  # (N,K,R)

                pcp_f_batch = pcp_batch.sum(axis=2)  # (N,J)
                pcp_b_batch = dev.np.einsum('njr,nkr -> njk', pcp_batch, der)  # (N,J,K)
                pcp_w_batch = dev.np.einsum('njr,nkr -> njk', pcp_batch, der*draws_batch)  # (N,J,K)
                ch_b_batch = dev.np.einsum('nr,nkr -> nk', pch_batch, der) # (N,K)
                ch_w_batch = dev.np.einsum('nr,nkr -> nk', pch_batch, der*draws_batch) # (N,K)
   
                pcp_f += dev.to_cpu(pcp_f_batch)
                pcp_b += dev.to_cpu(pcp_b_batch)
                pcp_w += dev.to_cpu(pcp_w_batch)
                ch_b += dev.to_cpu(ch_b_batch)
                ch_w += dev.to_cpu(ch_w_batch)

        pch = np.concatenate(pch, axis=-1) 
        # Log-likelihood
        lik = pch.mean(axis=1)  # (N, )
        loglik = np.log(lik)
        loglik = loglik if weights is None else loglik*weights
        loglik = loglik.sum()

        output = (-loglik, )

        # Gradient
        if return_gradient:
            lik = lik if panels is None else lik[panels]
            Rlik = R*lik[:, None]

            Xf = X[:, :, ~self._rvidx]  # (N,J,K)
            Xr = X[:, :, self._rvidx]  # (N,J,K)

            gr_f = (Xf*y).sum(axis=1) - dev.nan_safe_sum(Xf*pcp_f[..., None], axis=1)/Rlik
            gr_b = ((Xr*y).sum(axis=1)*ch_b - dev.nan_safe_sum(Xr*pcp_b, axis=1))/Rlik
            gr_w = ((Xr*y).sum(axis=1)*ch_w - dev.nan_safe_sum(Xr*pcp_w, axis=1))/Rlik

            # Put all gradients in a single array and aggregate them
            grad_n = self._concat_gradients(gr_f, gr_b, gr_w)  # (N,K)
            if weights is not None:
                grad_n = grad_n*weights[:, None]
            grad = grad_n.sum(axis=0)  # (K,)

            output += (-grad.ravel(), )
            output += (grad_n, )

        if not return_gradient or self.total_fun_eval == 0:
            self.total_fun_eval += 1
            if self.verbose > 1:
                print(f"Evaluation {self.total_fun_eval} Log-Lik.={-loglik:.2f}")
        return _unpack_tuple(output)

    def _concat_gradients(self, gr_f, gr_b, gr_w):
        N, Kf, Kr = len(gr_f), (~self._rvidx).sum(), self._rvidx.sum()
        gr = np.empty((N, Kf+2*Kr))
        gr[:, np.where(~self._rvidx)[0]] = gr_f
        gr[:, np.where(self._rvidx)[0]] = gr_b
        gr[:, len(self._rvidx):] = gr_w
        return gr

    def _prob_product_across_panels(self, prob, panels):
        if panels is not None:
            panel_change_idx = np.concatenate(([0], np.where(panels[:-1] != panels[1:])[0] + 1))
            prob = np.multiply.reduceat(prob, panel_change_idx)
        return prob  # (Np,R)

    def _apply_distribution(self, betas_random):
        """Apply the mixing distribution to the random betas."""
        for k, dist in enumerate(self._rvdist):
            if dist == 'ln':
                betas_random[:, k, :] = dev.np.exp(betas_random[:, k, :])
            elif dist == 'tn':
                betas_random[:, k, :] = betas_random[:, k, :] *\
                    (betas_random[:, k, :] > 0)
        return betas_random

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

    def _generate_halton_draws(self, sample_size, n_draws, n_vars, shuffle=False, drop=100, primes=None):
        """Generate Halton draws for multiple random variables using different primes as base"""
        if primes is None:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 71, 73, 79, 83, 89, 97, 101,
                      103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
                      199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311]
        
        def halton_seq(length, prime=3, shuffle=False, drop=100):
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
            if shuffle:
                np.random.shuffle(seq)
            return seq

        draws = [halton_seq(sample_size*n_draws, prime=primes[i % len(primes)],
                            shuffle=shuffle, drop=drop).reshape(sample_size, n_draws) for i in range(n_vars)]
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
            # Test a simple run of the log-likelihood to see if CuPy is working
            X = np.array([[2, 1], [1, 3], [3, 1], [2, 4], [2, 1], [2, 4]])
            y = np.array([0, 1, 0, 1, 0, 1])
            N, J, K, R = 3, 2, 2, 5

            betas = np.array([.1, .1, .1, .1])
            X_, y_ = X.reshape(N, J, K), y.reshape(N, J, 1)

            # Compute log likelihood using xlogit
            model = MixedLogit()
            model._rvidx,  model._rvdist = np.array([True, True]), np.array(['n', 'n'])
            draws = model._generate_halton_draws(N, R, K)  # (N,Kr,R)
            model._loglik_gradient(betas, X_, y_, None, draws, None, None,
                                   {'samples': N, 'draws': R}, return_gradient=False)

            print("{} GPU device(s) available. xlogit will use GPU processing".format(n_gpus))
            return True
        else:
            print("*** No GPU device found. Verify CuPy is properly installed")
            return False


    def _bfgs_optimization(self, betas, X, y, panels, draws, weights, avail, batch_sizes, maxiter, ftol, gtol=1e-6, step_tol=1e-10, num_hess=False):
        """BFGS optimization routine."""
        
        res, g, grad_n = self._loglik_gradient(betas, X, y, panels, draws, weights, avail, batch_sizes, return_gradient=True)
        Hinv = np.linalg.pinv(np.dot(grad_n.T, grad_n))
        current_iteration = 0
        convergence = False
        step_tol_failed = False
        while True:
            old_g = g.copy()

            d = -Hinv.dot(g)

            step = 2
            while True:
                step = step/2
                s = step*d
                resnew = self._loglik_gradient(betas + s, X, y, panels, draws, weights, avail, batch_sizes, return_gradient=False)
                if step > step_tol:
                    if resnew <= res or step < 1e-10:
                        betas = betas + s
                        resnew, gnew, grad_n = self._loglik_gradient(betas, X, y, panels, draws, weights, avail, batch_sizes, return_gradient=True)
                        break
                else:
                    step_tol_failed = True
                    break

            current_iteration += 1
            if step_tol_failed:
                convergence = False
                message = "Local search could not find a higher log likelihood value"
                break

            old_res = res
            res = resnew
            g = gnew

            if np.abs(np.dot(d, old_g)) < gtol:
                convergence = True
                message = "The gradients are close to zero"
                break

            if np.abs(res - old_res) < ftol:
                convergence = True
                message = "Succesive log-likelihood values within tolerance limits"
                break

            if current_iteration > maxiter:
                convergence = False
                message = "Maximum number of iterations reached without convergence"
                break

            delta_g = g - old_g

            Hinv = Hinv + (((s.dot(delta_g) + (delta_g[None, :].dot(Hinv)).dot(
                delta_g))*np.outer(s, s)) / (s.dot(delta_g))**2) - ((np.outer(
                    Hinv.dot(delta_g), s) + (np.outer(s, delta_g)).dot(Hinv)) /
                    (s.dot(delta_g)))
        if (num_hess):
            K = len(betas)
            H = np.zeros((K, K))
            for i in range(K):
                tempGrad = lambda x: self._loglik_gradient(x, X, y, panels, draws, weights, avail, batch_sizes, return_gradient=True)[1][i]
                tempHessRow = approx_fprime(betas, tempGrad, epsilon=1.4901161193847656e-08)
                # approx_fprime only handles scalars, so an anonymous function must be created
                # Explicit epsilon comes from scipy 1.8 defaults, which don't seem to be
                # present in earlier scipy versions
                H[i, :] = tempHessRow

            Hinv = np.linalg.inv(H)
        else:
            Hinv = np.linalg.inv(np.dot(grad_n.T, grad_n))

        return {'success': convergence, 'x': betas, 'fun': res, 'message': message,
                'hess_inv': Hinv, 'nit': current_iteration, 'grad_n':grad_n, 'grad':g}