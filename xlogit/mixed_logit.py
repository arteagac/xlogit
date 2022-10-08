"""Implements all the logic for mixed logit models."""

# pylint: disable=invalid-name
import scipy.stats
from ._choice_model import ChoiceModel, diff_nonchosen_chosen
from ._device import device as dev
from .multinomial_logit import MultinomialLogit
from ._optimize import _minimize, _numerical_hessian
import numpy as np

"""
Notations
---------
    N : Number of choice situations
    P : Number of observations per panel
    J : Number of alternatives
    K : Number of variables (Kf: fixed, Kr: random)
"""


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
            halton=True, verbose=1, batch_size=None, halton_opts=None, tol_opts=None, robust=False, num_hess=False,
            scale_factor=None, optim_method="BFGS", mnl_init=True):
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

        batch_size : int, default=None
            Size of batches used to avoid GPU memory overflow.
            
        scale_factor : array-like, shape (n_samples*n_alts, ), default=None
            Scaling variable used for non-linear models. For WTP models, this is usually the negative of 
            the price variable.
            
        optim_method : str, default='BFGS'
            Optimization method to use for model estimation. It can be `BFGS` or `L-BFGS-B`.
            For non-linear (WTP-like) models, `L-BFGS-B` is used by default.

        robust: bool, default=False
            Whether robust standard errors should be computed

        num_hess: bool, default=False
            Whether numerical hessian should be used for estimation of standard errors

        mnl_init: bool, default=True
            Whether to initialize coefficients using estimates from a multinomial logit
        Returns
        -------
        None.
        """
        # Handle array-like inputs by converting everything to numpy arrays
        X, y, varnames, alts, isvars, ids, weights, panels, avail, scale_factor \
            = self._as_array(X, y, varnames, alts, isvars, ids, weights, panels, avail, scale_factor)

        self._validate_inputs(X, y, alts, varnames, isvars, ids, weights)

        if mnl_init and init_coeff is None:
            # Initialize coefficients using a multinomial logit model
            mnl = MultinomialLogit()
            mnl.fit(X, y, varnames, alts, ids, isvars=isvars, weights=weights,
                    avail=avail, base_alt=base_alt, fit_intercept=fit_intercept)
            init_coeff = np.concatenate((mnl.coeff_, np.repeat(.1, len(randvars))))
            init_coeff = init_coeff if scale_factor is None else np.append(init_coeff, 1.)

        self._pre_fit(alts, varnames, isvars, base_alt, fit_intercept, maxiter)

        betas, X, y, panels, draws, weights, avail, Xnames, scale = \
            self._setup_input_data(X, y, varnames, alts, ids, randvars, isvars=isvars, weights=weights, avail=avail,
                                   panels=panels, init_coeff=init_coeff, random_state=random_state, n_draws=n_draws,
                                   halton=halton, verbose=verbose, predict_mode=False, halton_opts=halton_opts,
                                   scale_factor=scale_factor)

        tol = {'ftol': 1e-10, 'gtol': 1e-6}
        if tol_opts is not None:
            tol.update(tol_opts)

        Xd, scale_d, avail = diff_nonchosen_chosen(X, y, scale, avail)  # Setup Xd as Xij - Xi*
        fargs = (Xd, panels, draws, weights, avail, scale_d, batch_size)
        if scale_factor is not None:
            optim_method = "L-BFGS-B"
        optim_res = _minimize(self._loglik_gradient, betas, args=fargs, method=optim_method, tol=tol['ftol'],
                              options={'gtol': tol['gtol'], 'maxiter': maxiter, 'disp': verbose > 1})        

        coef_names = np.append(Xnames, np.char.add("sd.", Xnames[self._rvidx]))

        if scale_factor is not None:
            coef_names = np.append(coef_names, "_scale_factor")

        num_hess = num_hess if scale_factor is None else True
        if optim_method == "L-BFGS-B":
            optim_res['grad_n'] = self._loglik_gradient(optim_res['x'], *fargs, return_gradient=True)[2]

        if num_hess or optim_method == "L-BFGS-B":
            optim_res['hess_inv'] = _numerical_hessian(optim_res['x'], self._loglik_gradient, args=fargs)
        
        self._post_fit(optim_res, coef_names, X.shape[0], verbose, robust)



    def predict(self, X, varnames, alts, ids, isvars=None, weights=None, avail=None,  panels=None, random_state=None,
                n_draws=1000, halton=True, verbose=1, batch_size=None, return_proba=False, return_freq=False,
                halton_opts=None, scale_factor=None):
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
            Size of batches used to GPU avoid memory overflow. 
            
        scale_factor : array-like, shape (n_samples*n_alts, ), default=None
            Scaling variable used for non-linear WTP-like models. This is usually the negative of the price variable..

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
        X, _, varnames, alts, isvars, ids, weights, panels, avail, scale_factor \
            = self._as_array(X, None, varnames, alts, isvars, ids, weights, panels, avail, scale_factor)
        
        self._validate_inputs(X, None, alts, varnames, isvars, ids, weights)
        
        betas, X, _, panels, draws, weights, avail, Xnames, scale = \
            self._setup_input_data(X, None, varnames, alts, ids, self.randvars,  isvars=isvars, weights=weights,
                                   avail=avail, panels=panels, init_coeff=self.coeff_, random_state=random_state,
                                   n_draws=n_draws, halton=halton, verbose=verbose, predict_mode=True,
                                   halton_opts=halton_opts, scale_factor=scale_factor)
        
        coeff_names = np.append(Xnames, np.char.add("sd.", Xnames[self._rvidx]))
        coeff_names = coeff_names if scale_factor is None else np.append(coeff_names, "_scale_factor")
        if not np.array_equal(coeff_names, self.coeff_names):
            raise ValueError("The provided 'varnames' yield coefficient names that are inconsistent with the stored "
                             "in 'self.coeff_names'")
        
        
        lambdac = 1 if scale_factor is None else betas[-1]
        sca = 0 if scale_factor is None else scale[:, :, None]
        
        #=== 2. Compute choice probabilities
        Xf = X[:, :, ~self._rvidx]  # Data for fixed parameters
        Xr = X[:, :, self._rvidx]  # Data for random parameters
        betas, Xr, avail = dev.to_gpu(betas), dev.to_gpu(Xr), dev.to_gpu(avail)
        lambdac, sca = dev.to_gpu(lambdac), dev.to_gpu(sca)
        
        # Utility for fixed parameters
        Bf = betas[np.where(~self._rvidx)[0]]  # Fixed betas
        Vf = dev.np.einsum('njk,k -> nj', Xf, Bf)  # (N, J-1)
        
        proba = []  # Temp batching storage
        for batch_start, batch_end in batches_idx(batch_size, n_draws):
            draws_ = dev.to_gpu(draws[:, :, batch_start: batch_end])

            # Utility for random parameters 
            Br = self._transform_rand_betas(betas, draws_)  # Get random coefficients
            Vr = dev.cust_einsum("njk,nkr -> njr", Xr, Br)  # (N,J-1,R)
            
            eV = dev.np.exp(lambdac*(Vf[:, :, None] + Vr - sca))
            Vdr, Br = None, None # Release memory

            eV = eV if avail is None else eV*avail[:, :, None]  
            proba_ = eV/dev.np.sum(eV, axis=1, keepdims=True)  # (N,J,R)
            # proba_ = self._prob_product_across_panels(proba_, panels)  # (Np,J,R)

            proba.append(dev.to_cpu(proba_))

        proba = np.concatenate(proba, axis=-1) 
        proba = proba.mean(axis=-1)   # (N,J)
        
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
                          predict_mode=False, halton_opts=None, scale_factor=None):
        if random_state is not None:
            np.random.seed(random_state)

        self._check_long_format_consistency(ids, alts)
        y = self._format_choice_var(y, alts) if not predict_mode else None
        X, Xnames = self._setup_design_matrix(X)
        self._model_specific_validations(randvars, Xnames)

        N, J, K, R = X.shape[0], X.shape[1], X.shape[2], n_draws
        Kr, Ks = len(randvars), 1 if scale_factor is not None else 0

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

        if avail is not None:
            avail = avail.reshape(N, J)

        # Generate draws
        n_samples = N if panels is None else panels[-1] + 1
        draws = self._generate_draws(n_samples, R, halton, halton_opts=halton_opts)
        draws = draws if panels is None else draws[panels]  # (N,Kr,R)
      
        if weights is not None:  # Reshape weights to match input data
            weights = wights.reshape(N, J)[:, 0] 
            if panels is not None:
                panel_change_idx = np.concatenate(([0], np.where(panels[:-1] != panels[1:])[0] + 1))
                weights = weights[panel_change_idx]

        if init_coeff is None:
            betas = np.repeat(.1, K + Kr)
        else:
            betas = init_coeff
            if len(init_coeff) != (K + Kr + Ks):
                raise ValueError("The length of init_coeff must be: {}".format(K + Kr + Ks))
        
        scale = None if scale_factor is None else scale_factor.reshape(N, J)

        if dev.using_gpu and verbose > 0:
            print("GPU processing enabled.")
        return betas, X, y, panels, draws, weights, avail, Xnames, scale

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

    def _loglik_gradient(self, betas, Xd, panels, draws, weights, avail, scale_d, batch_size, return_gradient=True):
        """Compute the log-likelihood and gradient.

        Fixed and random parameters are handled separately to speed up the estimation and the results are concatenated.
        """
        N, R, Kr, Kf = Xd.shape[0], draws.shape[2], np.sum(self._rvidx), np.sum(~self._rvidx)
        
        lambdac = 1 if scale_d is None else betas[-1]
        Xd = Xd if scale_d is None else Xd*lambdac  # Multiply data by lambda coefficient when scaling is in use
        
        Xdf = Xd[:, :, ~self._rvidx]  # Data for fixed parameters
        Xdr = Xd[:, :, self._rvidx]  # Data for random parameters
        
        sca = 0 if scale_d is None else (lambdac*scale_d)[:, :, None]
        betas, Xdf, Xdr, avail, sca = dev.to_gpu(betas), dev.to_gpu(Xdf), dev.to_gpu(Xdr), dev.to_gpu(avail), dev.to_gpu(sca)

        # Utility for fixed parameters
        Bf = betas[np.where(~self._rvidx)[0]]  # Fixed betas
        Vdf = dev.np.einsum('njk,k -> nj', Xdf, Bf)  # (N, J-1)
        
        proba, gr_f, gr_u, gr_s, gr_l = [], np.zeros((N, Kf)), np.zeros((N, Kr)), np.zeros((N, Kr)), np.zeros((N, 1))  # Temp batching storage
        for batch_start, batch_end in batches_idx(batch_size, n_samples=R):
            draws_ = dev.to_gpu(draws[:, :, batch_start: batch_end])

            # Utility for random parameters 
            Br = self._transform_rand_betas(betas, draws_)  # Get random coefficients
            Vdr = dev.cust_einsum("njk,nkr -> njr", Xdr, Br)  # (N,J-1,R)
            
            eVd = dev.np.exp(Vdf[:, :, None] + Vdr - sca)
            Vdr, Br = None, None # Release memory
            eVd = eVd if avail is None else eVd*avail[:, :, None]  # Availablity of alts.
            # TODO: Handle availability
            proba_n = 1/(1+eVd.sum(axis=1)) # (N,R)
            proba_ = self._prob_product_across_panels(proba_n, panels) # (Np,R)
            
            if return_gradient:
                # The gradients are stored as a summation and at the end divided by R
                pprod = proba_*proba_ if panels is None else proba_[panels]*proba_n
                
                # For fixed coefficients
                dprod_f = -dev.np.einsum("njk,njr -> nkr", Xdf, eVd)  # (N,K,R)
                der_prod_f = dprod_f*pprod[:, None, :]     # (N,K,R)
                gr_f += dev.to_cpu((der_prod_f).sum(axis=2))  # (N,K)  
                
                # For random coefficients
                der = self._compute_derivatives(betas, draws_)  # (N,K,R)
                dprod_r = -dev.np.einsum("njk,njr -> nkr", Xdr, eVd)  # (N,K,R)
                der_prod_r = dprod_r*pprod[:, None, :]*der   # (N,K,R)
                gr_u += dev.to_cpu((der_prod_r).sum(axis=2))  # (N,K)
                gr_s += dev.to_cpu((der_prod_r*draws_).sum(axis=2))  # (N,K)
                
                # For WTP lambda scaling
                if scale_d is not None:
                    dprod_l = -dev.np.einsum("njr,njr -> nr", dev.np.log(eVd)/lambdac, eVd)[:, None, :] # (N,K,R)
                    der_prod_l = dprod_l*pprod[:, None, :]
                    gr_l += dev.to_cpu((der_prod_l).sum(axis=2))
                    
                
            proba_ = proba_.sum(axis=1)  # (N, )
            proba.append(dev.to_cpu(proba_))

        lik = np.stack(proba).sum(axis=0)/R  # (N, )
        loglik = np.log(lik) if weights is None else np.log(lik)*weights
        loglik = loglik.sum()
        output = (-loglik, )
        if return_gradient:
            lik = lik if panels is None else lik[panels]  # (N,)
            Rlik = R*lik[:, None]
            gr_f, gr_u, gr_s  = gr_f/Rlik, gr_u/Rlik, gr_s/Rlik
            grad_n = self._concat_gradients(gr_f, gr_u, gr_s)  # (N,K)
            grad_n = grad_n if scale_d is None else np.append(grad_n, gr_l/Rlik, 1)
            if weights is not None:
                weights = weights if panels is None else weights[panels]  # (N,)
                grad_n = grad_n*weights[:, None]
            grad = grad_n.sum(axis=0)
            output += (-grad, grad_n)

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
            idx = np.concatenate(([0], np.where(panels[:-1] != panels[1:])[0] + 1, [len(prob)]))
            prob = dev.np.vstack([prob[idx[i]:idx[i+1]].prod(axis=0) for i in range(len(idx) - 1)])
            
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
            betas_random = self._transform_rand_betas(betas, draws)
            for k, dist in enumerate(self._rvdist):
                if dist == 'ln':
                    der[:, k, :] = betas_random[:, k, :]
                elif dist == 'tn':
                    der[:, k, :] = 1*(betas_random[:, k, :] > 0)
        return der

    def _transform_rand_betas(self, betas, draws):
        """Compute the products between the betas and the random coefficients.

        This method also applies the associated mixing distributions
        """
        # Extract coeffiecients from betas array
        br_mean = betas[np.where(self._rvidx)[0]]
        br_sd = betas[len(self._rvidx):len(self._rvidx) + np.sum(self._rvidx)] 
        # Compute: betas = mean + sd*draws
        betas_random = br_mean[None, :, None] + draws*br_sd[None, :, None]
        betas_random = self._apply_distribution(betas_random)
        return betas_random

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
            y = np.array([0, 1, 0, 1, 0, 1]).astype(bool)
            N, J, K, R = 3, 2, 2, 5

            betas = np.array([.1, .1, .1, .1])
            Xd =  X[~y, :].reshape(N, J - 1, K) - X[y, :].reshape(N, 1, K) 

            # Compute log likelihood using xlogit
            model = MixedLogit()
            model._rvidx,  model._rvdist = np.array([True, True]), np.array(['n', 'n'])
            draws = model._generate_halton_draws(N, R, K)  # (N,Kr,R)
            model._loglik_gradient(betas, Xd, None, draws, None, None, None,
                                   batch_size=R, return_gradient=False)

            print("{} GPU device(s) available. xlogit will use GPU processing".format(n_gpus))
            return True
        else:
            print("*** No GPU device found. Verify CuPy is properly installed")
            return False
  
_unpack_tuple = lambda x : x if len(x) > 1 else x[0]
      
def batches_idx(batch_size, n_samples):
    batch_size = n_samples if batch_size is None else min(n_samples, batch_size)
    n_batches = n_samples//batch_size + int(n_samples % batch_size != 0)
    return [(batch*batch_size, batch*batch_size + batch_size) \
        for batch in range(n_batches)]

