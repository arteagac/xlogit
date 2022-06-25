"""Implements multinomial and mixed logit models."""
# pylint: disable=invalid-name

import numpy as np
from scipy.stats import t
from time import time
from abc import ABC
import warnings

"""
Notations
---------
    N : Number of choice situations
    P : Number of observations per panel
    J : Number of alternatives
    K : Number of variables (Kf: fixed, Kr: random)
"""


class ChoiceModel(ABC):
    """Base class for estimation of discrete choice models."""

    def __init__(self):
        self.coeff_names = None
        self.coeff_ = None
        self.stderr = None
        self.zvalues = None
        self.pvalues = None
        self.loglikelihood = None
        self.total_fun_eval = 0
        self.verbose = 1
        self.robust = False

    def _reset_attributes(self):
        self.coeff_names = None
        self.coeff_ = None
        self.stderr = None
        self.zvalues = None
        self.pvalues = None
        self.loglikelihood = None
        self.total_fun_eval = 0
        self.verbose = 1
        self.robust = False

    def _as_array(self, X, y, varnames, alts, isvars, ids, weights, panels,
                  avail, scale_factor):
        X = np.asarray(X)
        y = np.asarray(y)
        varnames = np.asarray(varnames) if varnames is not None else None
        alts = np.asarray(alts) if alts is not None else None
        isvars = np.asarray(isvars) if isvars is not None else None
        ids = np.asarray(ids) if ids is not None else None
        weights = np.asarray(weights) if weights is not None else None
        panels = np.asarray(panels) if panels is not None else None
        avail = np.asarray(avail) if avail is not None else None
        scale_factor = np.asarray(scale_factor) if scale_factor is not None else None
        return X, y, varnames, alts, isvars, ids, weights, panels, avail, scale_factor

    def _pre_fit(self, alts, varnames, isvars, base_alt,
                 fit_intercept, maxiter):
        self._reset_attributes()
        self._fit_start_time = time()
        self._isvars = [] if isvars is None else list(isvars)
        self._asvars = [v for v in varnames if v not in self._isvars]
        self._varnames = list(varnames)  # Easier to handle with lists
        self._fit_intercept = fit_intercept
        self.alternatives = np.sort(np.unique(alts))
        self.base_alt = self.alternatives[0] if base_alt is None else base_alt
        self.maxiter = maxiter

    def _post_fit(self, optim_res, coeff_names, sample_size, verbose=1, robust=False):
        self.convergence = optim_res['success']
        self.coeff_ = optim_res['x']
        self.hess_inv = optim_res['hess_inv']
        self.covariance = self._robust_covariance(optim_res['hess_inv'], optim_res['grad_n']) \
            if robust else optim_res['hess_inv']
        self.stderr = np.sqrt(np.diag(self.covariance))
        self.zvalues = self.coeff_/self.stderr
        self.pvalues = 2*t.pdf(-np.abs(self.zvalues), df=sample_size)
        self.loglikelihood = -optim_res['fun']
        self.estimation_message = optim_res['message']
        self.coeff_names = coeff_names
        self.total_iter = optim_res['nit']
        self.estim_time_sec = time() - self._fit_start_time
        self.sample_size = sample_size
        self.aic = 2*len(self.coeff_) - 2*self.loglikelihood
        self.bic = np.log(sample_size)*len(self.coeff_) - 2*self.loglikelihood
        self.grad_n = optim_res['grad_n']
        self.total_fun_eval = optim_res['nfev']


        if not self.convergence and verbose > 0:
            print("**** The optimization did not converge after {} "
                  "iterations. ****".format(self.total_iter))
            print("Message: "+optim_res['message'])


    def _robust_covariance(self, hess_inv, grad_n):

        """ Estimates the robust covariance matrix.

        This follows the methodology lined out in p.486-488 in the Stata 16 reference manual.
        Benchmarked against Stata 17.
        """
        n = np.shape(grad_n)[0]
        grad_n_sub = grad_n-(np.sum(grad_n, axis=0)/n) #subtract out mean gradient value
        inner = np.transpose(grad_n_sub)@grad_n_sub
        correction = ((n)/(n-1))
        covariance = correction*(hess_inv@inner@hess_inv)
        return covariance

    def _setup_design_matrix(self, X):
        """Setups and reshapes input data after adding isvars and intercept.

        Setup the design matrix by adding the intercept when necessary and
        converting the isvars to a dummy representation that removes the base
        alternative.
        """
        J = len(self.alternatives)
        N = int(len(X)/J)
        isvars = self._isvars.copy()
        asvars = self._asvars.copy()
        varnames = self._varnames.copy()

        if self._fit_intercept:
            isvars.insert(0, '_intercept')
            varnames.insert(0, '_intercept')
            X = np.hstack((np.ones(J*N)[:, None], X))

        ispos = [varnames.index(i) for i in isvars]  # Position of IS vars
        aspos = [varnames.index(i) for i in asvars]  # Position of AS vars

        # Create design matrix
        # For individual specific variables
        if isvars:
            # Create a dummy individual specific variables for the alt
            dummy = np.tile(np.eye(J), reps=(N, 1))
            # Remove base alternative
            dummy = np.delete(dummy,
                              np.where(self.alternatives == self.base_alt)[0],
                              axis=1)
            Xis = X[:, ispos]
            # Multiply dummy representation by the individual specific data
            Xis = np.einsum('nj,nk->njk', Xis, dummy)
            Xis = Xis.reshape(N, J, (J-1)*len(ispos))

        # For alternative specific variables
        if asvars:
            Xas = X[:, aspos]
            Xas = Xas.reshape(N, J, -1)

        # Set design matrix based on existance of asvars and isvars
        if asvars and isvars:
            X = np.dstack((Xis, Xas))
        elif asvars:
            X = Xas
        elif isvars:
            X = Xis

        names = ["{}.{}".format(isvar, j) for isvar in isvars
                 for j in self.alternatives if j != self.base_alt] + asvars
        names = np.array(names)

        return X, names

    def _check_long_format_consistency(self, ids, alts):
        """Ensure that data in long format is consistent.

        It raises an error if the array of alternative indexes is incomplete
        """
        uq_alts, idx = np.unique(alts, return_index=True)
        uq_alts = uq_alts[np.argsort(idx)]
        expected_alts = np.tile(uq_alts, int(len(ids)/len(uq_alts)))
        if not np.array_equal(alts, expected_alts):
            raise ValueError('inconsistent alts values in long format')
        _, obs_by_id = np.unique(ids, return_counts=True)
        if not np.all(obs_by_id/len(uq_alts)):  # Multiple of J
            raise ValueError('inconsistent alts and ids values in long format')

    def _format_choice_var(self, y, alts):
        """Format choice (y) variable as one-hot encoded."""
        uq_alts = np.unique(alts)
        J, N = len(uq_alts), len(y)//len(uq_alts)
        # When already one-hot encoded the sum by row is one
        if isinstance(y[0], (np.number, np.bool_)) and \
            np.array_equal(y.reshape(N, J).sum(axis=1), np.ones(N)):
            return y
        else:
            y1h = (y == alts).astype(int)  # Apply one hot encoding
            if np.array_equal(y1h.reshape(N, J).sum(axis=1), np.ones(N)):
                return y1h
            else:
                raise ValueError("inconsistent 'y' values. Make sure the "
                                 "data has one choice per sample")

    def _validate_inputs(self, X, y, alts, varnames, isvars, ids, weights):
        """Validate potential mistakes in the input data."""
        if varnames is None:
            raise ValueError('The parameter varnames is required')
        if alts is None:
            raise ValueError('The parameter alternatives is required')
        if X.ndim != 2:
            raise ValueError("X must be an array of two dimensions in "
                             "long format")
        if y is not None and y.ndim != 1:
            raise ValueError("y must be an array of one dimension in "
                             "long format")
        if len(varnames) != X.shape[1]:
            raise ValueError("The length of varnames must match the number "
                             "of columns in X")

    def summary(self):
        """Show estimation results in console."""
        if self.coeff_ is None:
            warnings.warn("The current model has not been yet estimated",
                          UserWarning)
            return
        if not self.convergence:
            warnings.warn("WARNING: Convergence not reached. The estimates may not be reliable.",
                          UserWarning)
        if self.convergence:
            print("Optimization terminated successfully.")

        print("    Message: {}".format(self.estimation_message ))
        print("    Iterations: {}".format(self.total_iter))
        print("    Function evaluations: {}".format(self.total_fun_eval))
        print("Estimation time= {:.1f} seconds".format(self.estim_time_sec))
        print("-"*75)
        print("{:19} {:>13} {:>13} {:>13} {:>13}"
              .format("Coefficient", "Estimate", "Std.Err.", "z-val", "P>|z|"))
        print("-"*75)
        fmt = "{:19} {:13.7f} {:13.7f} {:13.7f} {:13.3g} {:3}"
        for i in range(len(self.coeff_)):
            signif = ""
            if self.pvalues[i] < 0.001:
                signif = "***"
            elif self.pvalues[i] < 0.01:
                signif = "**"
            elif self.pvalues[i] < 0.05:
                signif = "*"
            elif self.pvalues[i] < 0.1:
                signif = "."
            print(fmt.format(self.coeff_names[i][:19], self.coeff_[i],
                             self.stderr[i], self.zvalues[i], self.pvalues[i],
                             signif))
        print("-"*75)
        print("Significance:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print("")
        print("Log-Likelihood= {:.3f}".format(self.loglikelihood))
        print("AIC= {:.3f}".format(self.aic))
        print("BIC= {:.3f}".format(self.bic))


def diff_nonchosen_chosen(X, y, scale, avail):
    # Setup Xd as Xij - Xi* (difference between non-chosen and chosen alternatives)
    N, J, K = X.shape
    X, y = X.reshape(N*J, K), y.astype(bool).reshape(N*J, )
    Xd =  X[~y, :].reshape(N, J - 1, K) - X[y, :].reshape(N, 1, K)
    scale = scale.reshape(N*J, ) if scale is not None else None
    scale_d = scale[~y].reshape(N, J - 1) - scale[y].reshape(N, 1) if scale is not None else None
    avail = avail.reshape(N*J)[~y].reshape(N, J - 1) if avail is not None else None
    return Xd, scale_d, avail