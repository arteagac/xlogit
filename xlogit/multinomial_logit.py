"""
Implements multinomial and conditional logit models
"""
# pylint: disable=line-too-long,invalid-name

import numpy as np
from ._choice_model import ChoiceModel


class MultinomialLogit(ChoiceModel):
    """Class for estimation of Multinomial and Conditional Logit Models"""

    def fit(self, X, y, varnames=None, alt=None, isvars=None, id=None,
            weights=None, base_alt=None, fit_intercept=False, init_coeff=None,
            maxiter=2000, random_state=None, verbose=1):

        X, y, varnames, alt, isvars, id, weights, _\
            = self._as_array(X, y, varnames, alt, isvars, id, weights, None)
        self._validate_inputs(X, y, alt, varnames, isvars, id, weights, None,
                              base_alt, fit_intercept, maxiter)

        self._pre_fit(alt, varnames, isvars, base_alt, fit_intercept, maxiter)
        X, y, panel = self._arrange_long_format(X, y, id, alt)

        if random_state is not None:
            np.random.seed(random_state)

        if init_coeff is None:
            betas = np.repeat(.0, X.shape[2])
        else:
            betas = init_coeff
            if len(init_coeff) != X.shape[1]:
                raise ValueError("The size of initial_coeff must be: "
                                 + int(X.shape[1]))

        X, Xnames = self._setup_design_matrix(X)
        y = y.reshape(X.shape[0], X.shape[1])

        # Call optimization routine
        optimizat_res = self._bfgs_optimization(betas, X, y, weights, maxiter)
        self._post_fit(optimizat_res, Xnames, X.shape[0], verbose)

    def _compute_probabilities(self, betas, X):
        XB = X.dot(betas)
        eXB = np.exp(XB)
        p = eXB/np.sum(eXB, axis=1, keepdims=True)  # (N,J)
        return p

    def _loglik_and_gradient(self, betas, X, y, weights):
        p = self._compute_probabilities(betas, X)
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

    def _bfgs_optimization(self, betas, X, y, weights, maxiter):
        res, g, Hinv = self._loglik_and_gradient(betas, X, y, weights)
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
                                                            weights)
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
