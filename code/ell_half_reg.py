import numpy as np
import cvxpy as cp
import numpy.linalg as LA
import dccp
import pdb

# Manual implementation of CCP
def ccp_noncvx(A, b, lmbda, max_iter=40, eps=1e-10, x0=None):
    n = A.shape[1]
    x = cp.Variable(n)

    # initialize with solution from Tikhonov regularization
    if x0 is not None:
        x.value = x0
    else:
        min_reg = np.max((lmbda, 1e-5))
        I = np.eye(n)
        x.value = LA.solve(A.T @ A + min_reg * I, A.T @ b)
    
    tk = cp.Parameter(n, nonneg=True)
    tk.value = np.sqrt(np.abs(x.value))

    obj = cp.sum_squares(A @ x - b) + lmbda * cp.norm1(x / (2 * tk))
    prob = cp.Problem(cp.Minimize(obj))

    for iter in range(max_iter):
        tk.value = np.maximum(0.5 * (np.abs(x.value)/tk.value + tk.value),eps)
        prob.solve(solver=cp.CLARABEL, verbose=False)

    return x.value

# Solving the problem with DCCP
def DCCP_ell_half(A, b, lmbda, x0=None):
    n = A.shape[1]
    x = cp.Variable(n)
    t = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + lmbda * cp.sum(t)
    constraints = [t >= 0, t ** 2 >= cp.abs(x)]
    prob = cp.Problem(cp.Minimize(obj), constraints)

    if x0 is not None:
        x.value = x0
    else:
        min_reg = np.max((lmbda, 1e-5))
        I = np.eye(n)
        x.value = LA.solve(A.T @ A + min_reg * I, A.T @ b)

    t.value = np.sqrt(np.abs(x.value))
    prob.solve(method = "dccp", solver = cp.CLARABEL, verbose=False, tau_max=500, tau=500) 
    return x.value