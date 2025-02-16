
import cvxpy as cp
from cvxpy import Variable, Parameter, Minimize, Problem
import numpy as np
from numpy import ones
from matplotlib import pyplot as plt


def get_problem(n, A, b):
    x = Variable(n)
    xk = Parameter(n)
    obj = Minimize((ones(n) - 2 * xk) @ x)
    constr = [A @ x <= b, 0 <= x, x <= 1]
    return x, xk, Problem(obj, constr)

def ccp(x, xk, problem, K):
    xk_ = ones((K+1, x.size)) / 2
    for i in range(K):
        xk.value = xk_[i]
        problem.solve()
        xk_[i+1] = x.value
    return xk_

def plot_sol(x_):
    plt.imshow(x_.T, cmap='gray', aspect='auto')
    plt.xticks(np.arange(len(x_)))
    plt.xlabel('k')
    plt.yticks([])
    plt.colorbar()
    for v in np.arange(0.5, len(x_) - 0.5):
        plt.axvline(v, color=np.ones(3)/3, linewidth=0.8)
    

# small problem

n = 4

A = np.array([[-1, -1, 1, 0],
              [1, 1, 0, -1],
              [0, 1, -1, 1]])

b = np.array([0, 1, 1])

x, xk, problem = get_problem(n, A, b)
K = 3
xk_ = ccp(x, xk, problem, K)

plt.figure(figsize=(4, 3))
plot_sol(xk_)
plt.savefig('img/sat_small.pdf', bbox_inches='tight')
plt.show()


# large problem

m, n = 120, 40

np.random.seed(4)

A = np.zeros((m, n))
for i in range(m):
    idx = np.random.choice(n, 3, replace=False)
    A[i, idx] = np.random.choice([-1, 1], 3)
    
b = np.sum(A == 1, axis=1) - np.ones(m)

x, xk, problem = get_problem(n, A, b)
K = 11
xk_ = ccp(x, xk, problem, K)
val = np.sum(xk_ - xk_**2, axis=1)

plt.figure(figsize=(4, 3))
plot_sol(xk_)
plt.savefig('img/sat_large.pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(4, 3))
plt.plot(val, 'blue', linewidth=2)
plt.xticks(np.arange(K + 1))
plt.xlabel('k')
plt.ylim([0, max(val)*1.05])
plt.savefig('img/sat_large_value.pdf', bbox_inches='tight')
plt.show()
