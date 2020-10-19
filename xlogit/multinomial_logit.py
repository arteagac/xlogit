"""
Implements multinomial and conditional logit models
"""
# pylint: disable=line-too-long,invalid-name

import numpy as np
from ._choice_model import ChoiceModel


class MultinomialLogit(ChoiceModel):
    """Class for estimation of Multinomial and Conditional Logit Models"""

    def fit(self, X, y, varnames=None, alt=None, isvars=None,
            base_alt=None, fit_intercept=False, init_coeff=None, maxiter=2000,
            random_state=None, verbose=1):
        """
        Fits multinomial model using the given data parameters.

        Parameters
        ----------
        X: numpy array, shape [n_samples, n_features], Data matrix in long
        format.
        y: numpy array, shape [n_samples, ], Vector of choices or discrete
        output.
        alternatives: list, List of alternatives names or codes.
        isvars: list, List of individual specific variables
        base_alt: string, base alternative. When not specified, pymlogit uses
        the first alternative in alternatives vector by default.
        max_iterations: int, Maximum number of optimization iterations
        fit_intercept: bool
        """
        self._validate_inputs(X, y, alt, varnames, isvars,
                              base_alt, fit_intercept, maxiter)

        self._pre_fit(alt, varnames, isvars, base_alt,
                      fit_intercept, maxiter)

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
        optimizat_res = self._bfgs_optimization(betas, X, y, maxiter)
        self._post_fit(optimizat_res, Xnames, X.shape[0], verbose)

    def _compute_probabilities(self, betas, X):
        XB = X.dot(betas)
        eXB = np.exp(XB)
        p = eXB/np.sum(eXB, axis=1, keepdims=True)
        return p

    def _loglik_and_gradient(self, betas, X, y):
        """
        Computes the log likelihood of the parameters B with self.X
        and self.y data

        Parameters
        ----------
                B: numpy array, shape [n_parameters, ], Vector of betas
                or parameters

        Returns
        ----------
                ll: float, Optimal value of log likelihood (negative)
                g: numpy array, shape[n_parameters], Vector of individual
                gradients (negative)
                Hinv: numpy array, shape [n_parameters, n_parameters]
                Inverse of the approx. hessian
        """
        p = self._compute_probabilities(betas, X)
        # Log likelihood
        lik = np.sum(y*p, axis=1)
        loglik = np.sum(np.log(lik))
        # Individual contribution to the gradient
        g_i = np.einsum('nj,njk -> nk', (y-p), X)

        H = np.dot(g_i.T, g_i)
        Hinv = np.linalg.inv(H)
        grad = np.sum(g_i, axis=0)
        return -loglik, -grad, Hinv

    def _bfgs_optimization(self, betas, X, y, maxiter):
        """
        Performs the BFGS optimization routine. For more information in this
        newton-based optimization technique see:
        http://aria42.com/blog/2014/12/understanding-lbfgs

        Parameters
        ----------
                betas: numpy array, shape [n_parameters, ], Vector of betas
                or parameters

        Returns
        ----------
                B: numpy array, shape [n_parameters, ], Optimized parameters
                res: float, Optimal value of optimization function
                (log likelihood in this case)
                g: numpy array, shape[n_parameters], Vector of individual
                gradients (negative)
                Hinv: numpy array, shape [n_parameters, n_parameters]
                Inverse of the approx. hessian
                convergence: bool, True when optimization converges
                current_iteration: int, Iteration when convergence was reached
        """
        res, g, Hinv = self._loglik_and_gradient(betas, X, y)
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
                resnew, gnew, _ = self._loglik_and_gradient(betas, X, y)
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
