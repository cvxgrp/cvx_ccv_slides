import os
import numpy as np 
import cvxpy as cp 
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from ell_half_reg import ccp_noncvx, DCCP_ell_half
from matplotlib.ticker import ScalarFormatter

img_path = 'img'
if os.path.basename(os.getcwd()) == 'code':
    img_path = os.path.join('..', img_path)

# generate random data
ratio = 2
np.random.seed(7)
n = 20
m = int(ratio * n)
true_density = 1
threshold = 1e-5
x0 = 2*np.random.randn(n)
x0[np.random.rand(n) < 1 - true_density] = 0
A = np.random.randn(m, n)
b = A @ x0 
b += 0.01*LA.norm(b)*np.random.randn(m)

lmbda_max = 2 * LA.norm(A.T @ b, np.inf)
lmbda_upper = 5 * lmbda_max
lmbda_lower = 1e-3 * lmbda_max
lmbdas = np.logspace(np.log10(lmbda_lower), np.log10(lmbda_upper),
                      150, base=10)
x_lasso, nnz_lasso, x_noncvx, nnz_noncvx = [], [], [], []

# -----------------------------------------------------------------
#           compute Lasso regularization path
# -----------------------------------------------------------------
x = cp.Variable(n)
lmbda_var = cp.Parameter(nonneg=True)
prob_lasso = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) +  
                                    lmbda_var*cp.norm(x, 1))) 

for lmbda in lmbdas:
    lmbda_var.value = lmbda
    prob_lasso.solve(solver=cp.CLARABEL)
    x_lasso.append(x.value)
    nnz_lasso.append(np.sum(np.abs(x.value) >= threshold))

x_lasso = np.array(x_lasso) 
fit_lasso = [LA.norm(A @ x - b) ** 2 for x in x_lasso] 

# -----------------------------------------------------------------
#           compute ell-half regularization path
# -----------------------------------------------------------------
x_init=None

for lmbda in lmbdas:
    lmbda_var.value = lmbda
    x_without_DPP = ccp_noncvx(A, b, lmbda, max_iter=20, x0=x_init)

    # can also compute it using DCCP
    # x_with_DCCP = DCCP_ell_half(A, b, lmbda, x0=x_init)

    #print(LA.norm(x_without_DPP - x_with_DCCP))

    x_noncvx.append(x_without_DPP)
    x_init = x_without_DPP
    nnz_noncvx.append(np.sum(np.abs(x_without_DPP) >= threshold))

x_noncvx = np.array(x_noncvx)
fit_noncvx = [LA.norm(A @ x - b) ** 2 for x in x_noncvx]

# -----------------------------------------------------------------
#            plot cardinality and fit
# -----------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(lmbdas, nnz_lasso, 'b-')
ax1.plot(lmbdas, nnz_noncvx, 'r-')
ax1.set_xscale('log')
ax1.set_xlabel(r'$\lambda$', fontsize=25)
ax1.set_ylabel(r'$\text{card}(x)$', fontsize=30)
ax1.tick_params(axis='y', labelcolor='black', labelsize=20)
ax1.tick_params(axis='x', labelcolor='black', labelsize=20)

ax2 = ax1.twinx()
ax2.plot(lmbdas, fit_lasso, 'b--')
ax2.plot(lmbdas, fit_noncvx, 'r--')
ax2.set_ylabel(r'$\|Ax - b\|^2$', fontsize=30)
ax2.tick_params(axis='y', labelcolor='black', labelsize=20)

# Force scientific notation with a scale of 10^3
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((2, 2))
ax2.yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig(os.path.join(img_path, "card_and_fit.pdf"), dpi=300)

# -----------------------------------------------------------------
#            plot regularization path for lasso
# -----------------------------------------------------------------
fig = plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(lmbdas, x_lasso[:, i], label=f'$x_{i+1}$', lw=2, alpha=0.8)
plt.xscale('log')
plt.xlabel(r'$\lambda$', fontsize=25)
plt.ylabel(r'$x_i$', fontsize=30)
plt.ylim(-4, 4.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(axis='both', labelsize=22)
plt.tight_layout()
plt.savefig(os.path.join(img_path, "lasso_reg_path_larger.pdf"), dpi=300)


# -----------------------------------------------------------------
#            plot regularization path for ell-half
# -----------------------------------------------------------------
fig = plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(lmbdas, x_noncvx[:, i], label=f'$x_{i+1}$', lw=2, alpha=0.8)
plt.xscale('log')
plt.xlabel(r'$\lambda$', fontsize=25)
plt.ylabel(r'$x_i$', fontsize=30)
plt.ylim(-4, 4.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(axis='both', labelsize=22)
plt.tight_layout()
plt.savefig(os.path.join(img_path, "ell_half_reg_path_larger.pdf"), dpi=300)

# -----------------------------------------------------------------
#            plot scatter plot of fit vs cardinality
# -----------------------------------------------------------------
fig = plt.figure(figsize=(10, 6))
plt.scatter(nnz_lasso, fit_lasso, color='b', s=90, alpha=0.6)
plt.scatter(nnz_noncvx, fit_noncvx, color='r', s=60, alpha=0.6)
plt.xlabel(r'$\text{card}(x)$', fontsize=25)
plt.ylabel(r'$\|Ax - b\|^2$', fontsize=30)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tick_params(axis='both', labelsize=22)
plt.tight_layout()
plt.savefig(os.path.join(img_path, "scatter_fit_card_larger.pdf"), dpi=300)


# -----------------------------------------------------------------
#            plot pareto optimal points
# -----------------------------------------------------------------
fig = plt.figure(figsize=(10, 7))

nnz_lasso = np.array(nnz_lasso)
nnz_noncvx = np.array(nnz_noncvx)
fit_lasso = np.array(fit_lasso)
fit_noncvx = np.array(fit_noncvx)

unique_nnz_lasso = np.unique(nnz_lasso)
min_fit_lasso = np.array([np.min(fit_lasso[nnz_lasso == val]) for val in unique_nnz_lasso])

unique_nnz_noncvx = np.unique(nnz_noncvx)
min_fit_noncvx  = np.array([np.min(fit_noncvx[nnz_noncvx  == val]) for val in unique_nnz_noncvx ])

plt.plot(unique_nnz_lasso, min_fit_lasso, color='b', alpha=0.6, linestyle='-', marker='o', markersize=10)
plt.plot(unique_nnz_noncvx, min_fit_noncvx, color='r', alpha=0.6, linestyle='-', marker='o', markersize=10)


# Force scientific notation with a scale of 10^3
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((2, 2))
plt.gca().yaxis.set_major_formatter(formatter)

ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.xlabel(r'$\text{card}(x)$', fontsize=25)
plt.ylabel(r'$\|Ax - b\|^2$', fontsize=30)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(axis='both', labelsize=22)
plt.tight_layout()
plt.savefig(os.path.join(img_path, "pareto_curve_version2.pdf"), dpi=300)
