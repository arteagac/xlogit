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

    def _reset_attributes(self):
        self.coeff_names = None
        self.coeff_ = None
        self.stderr = None
        self.zvalues = None
        self.pvalues = None
        self.loglikelihood = None
        self.total_fun_eval = 0
        self.verbose = 1

    def _as_array(self, X, y, varnames, alts, isvars, ids, weights, panels,
                  avail):
        X = np.asarray(X)
        y = np.asarray(y)
        varnames = np.asarray(varnames) if varnames is not None else None
        alts = np.asarray(alts) if alts is not None else None
        isvars = np.asarray(isvars) if isvars is not None else None
        ids = np.asarray(ids) if ids is not None else None
        weights = np.asarray(weights) if weights is not None else None
        panels = np.asarray(panels) if panels is not None else None
        avail = np.asarray(avail) if avail is not None else None
        return X, y, varnames, alts, isvars, ids, weights, panels, avail

    def _pre_fit(self, alts, varnames, isvars, base_alt,
                 fit_intercept, maxiter):
        self._reset_attributes()
        self._fit_start_time = time()
        self._isvars = [] if isvars is None else list(isvars)
        self._asvars = [v for v in varnames if v not in self._isvars]
        self._varnames = list(varnames)  # Easier to handle with lists
        self._fit_intercept = fit_intercept
        self._alternatives = np.unique(alts)
        self.base_alt = self._alternatives[0] if base_alt is None else base_alt
        self.maxiter = maxiter

    def _post_fit(self, optimization_res, coeff_names, sample_size, verbose=1):
        self.convergence = optimization_res['success']
        self.coeff_ = optimization_res['x']
        self.stderr = np.sqrt(np.diag(optimization_res['hess_inv']))
        self.zvalues = self.coeff_/self.stderr
        self.pvalues = 2*t.pdf(-np.abs(self.zvalues), df=sample_size)
        self.loglikelihood = -optimization_res['fun']
        self.coeff_names = coeff_names
        self.total_iter = optimization_res['nit']
        self.estim_time_sec = time() - self._fit_start_time
        self.sample_size = sample_size
        self.aic = 2*len(self.coeff_) - 2*self.loglikelihood
        self.bic = np.log(sample_size)*len(self.coeff_) - 2*self.loglikelihood

        if not self.convergence and verbose > 0:
            print("**** The optimization did not converge after {} "
                  "iterations. ****".format(self.total_iter))
            print("Message: "+optimization_res['message'])

    def _setup_design_matrix(self, X):
        """Setups and reshapes input data after adding isvars and intercept.

        Setup the design matrix by adding the intercept when necessary and
        converting the isvars to a dummy representation that removes the base
        alternative.
        """
        J = len(self._alternatives)
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
                              np.where(self._alternatives == self.base_alt)[0],
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
                 for j in self._alternatives if j != self.base_alt] + asvars
        names = np.array(names)

        return X, names

    def _check_long_format_consistency(self, ids, alts, sorted_idx):
        """Ensure that data in long format is consistent.

        It raises an error if the array of alternative indexes is incomplete
        """
        alts = alts[sorted_idx]
        uq_alts = np.unique(alts)
        expected_alts = np.tile(uq_alts, int(len(ids)/len(uq_alts)))
        if not np.array_equal(alts, expected_alts):
            raise ValueError('inconsistent alts values in long format')
        _, obs_by_id = np.unique(ids, return_counts=True)
        if not np.all(obs_by_id/len(uq_alts)):  # Multiple of J
            raise ValueError('inconsistent alts and ids values in long format')

    def _arrange_long_format(self, X, y, ids, alts, panels=None):
        """Sort the input data for easy reshaping in future steps.

        This ensures that that data can be safely reshaped later to do
        everything in terms of matrix products.
        """
        if ids is not None:
            pnls = panels if panels is not None else np.ones(len(ids))
            alts = alts.astype(str)
            alts = alts if len(alts) == len(ids)\
                else np.tile(alts, int(len(ids)/len(alts)))
            cols = np.zeros(len(ids),
                            dtype={'names': ['panel', 'id', 'alt'],
                                   'formats': ['<f4', '<f4', '<U64']})
            cols['panel'], cols['id'], cols['alt'] = pnls, ids, alts
            sorted_idx = np.argsort(cols, order=['panel', 'id', 'alt'])
            X, y = X[sorted_idx], y[sorted_idx]
            if panels is not None:
                panels = panels[sorted_idx]
            self._check_long_format_consistency(ids, alts, sorted_idx)
        return X, y, panels

    def _validate_inputs(self, X, y, alts, varnames, isvars, ids, weights,
                         base_alt, fit_intercept, maxiter):
        """Validate potential mistakes in the input data."""
        if varnames is None:
            raise ValueError('The parameter varnames is required')
        if alts is None:
            raise ValueError('The parameter alternatives is required')
        if X.ndim != 2:
            raise ValueError("X must be an array of two dimensions in "
                             "long format")
        if y.ndim != 1:
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
            warnings.warn("WARNING: Convergence was not reached during"
                          "The given estimates may not be reliable",
                          UserWarning)
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
