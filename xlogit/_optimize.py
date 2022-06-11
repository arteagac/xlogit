import numpy as np
from scipy.optimize import minimize, approx_fprime

def _bfgs(loglik_fn, x, args, maxiter=2000, tol=1e-10, gtol=1e-6, step_tol=1e-10, disp=False):
    """BFGS optimization routine."""
    
    res, g, grad_n = loglik_fn(x, *args, **{'return_gradient': True})
    Hinv = np.linalg.pinv(np.dot(grad_n.T, grad_n))
    convergence = False
    step_tol_failed = False
    nit, nfev, njev = 0, 1, 1
    while True:
        old_g = g

        d = -Hinv.dot(g)

        step = 2
        while True:
            step = step/2
            s = step*d
            resnew = loglik_fn(x + s, *args, **{'return_gradient': False})
            nfev += 1
            if step > step_tol:
                if resnew <= res or step < step_tol:
                    x = x + s
                    resnew, gnew, grad_n = loglik_fn(x, *args, **{'return_gradient': True})
                    njev += 1
                    break
            else:
                step_tol_failed = True
                break

        nit += 1

        if step_tol_failed:
            convergence = False
            message = "Local search could not find a higher log likelihood value"
            break
        
        old_res = res
        res = resnew
        g = gnew
        gproj = np.abs(np.dot(d, old_g))
        
        if disp:
            print(f"Iteration: {nit} \t Log-Lik.= {resnew:.3f} \t |proj g|= {gproj:e}")

        if gproj < gtol:
            convergence = True
            message = "The gradients are close to zero"
            break

        if np.abs(res - old_res) < tol:
            convergence = True
            message = "Succesive log-likelihood values within tolerance limits"
            break

        if nit > maxiter:
            convergence = False
            message = "Maximum number of iterations reached without convergence"
            break

        delta_g = g - old_g

        Hinv = Hinv + (((s.dot(delta_g) + (delta_g[None, :].dot(Hinv)).dot(
            delta_g))*np.outer(s, s)) / (s.dot(delta_g))**2) - ((np.outer(
                Hinv.dot(delta_g), s) + (np.outer(s, delta_g)).dot(Hinv)) /
                (s.dot(delta_g)))

    Hinv = np.linalg.pinv(np.dot(grad_n.T, grad_n))
    return {'success': convergence, 'x': x, 'fun': res, 'message': message,
            'hess_inv': Hinv, 'grad_n':grad_n, 'grad':g, 'nit': nit, 'nfev': nfev, 'njev': njev}
    
def _minimize(loglik_fn, x, args, method, tol, options):
    if method == "BFGS":
        return _bfgs(loglik_fn, x, args=args, tol=tol, **options)
    elif method == "L-BFGS-B":
        return minimize(loglik_fn, x, args=args, jac=True, method='L-BFGS-B', tol=tol, options=options)
    else:
        raise ValueError(f"Unknown optimization method: {method}")
        
def _numerical_hessian(x, fn, args):
    H = np.empty((len(x), len(x)))
    eps = 1.4901161193847656e-08 # From scipy 1.8 defaults
    
    for i in range(len(x)):
        fn_call = lambda x_: fn(x_, *args)[1][i]
        hess_row = approx_fprime(x, fn_call, epsilon=eps)
        H[i, :] = hess_row
    
    Hinv = np.linalg.inv(H)
    return Hinv
