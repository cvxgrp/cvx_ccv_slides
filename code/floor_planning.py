import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


PLOT_ITERATIONS = True


img_path = 'img'
if os.path.basename(os.getcwd()) == 'code':
    img_path = os.path.join('..', img_path)


def draw(ax, k, x, y, c, val):

    a = ax[k] #ax[k//2, k%2]

    for (i, j) in E:
        a.plot([c[i][0], c[j][0]], [c[i][1], c[j][1]], 'r--', linewidth=0.5)

    for i in range(n):
        a.add_patch(plt.Rectangle((x[i], y[i]), w[i], h[i], fill=False))
        a.text(c[i][0], c[i][1], str(i+1), color='black', ha='center', va='center', fontsize=16)

    # Set the limits and aspect ratio
    a.set_xlim(-3.5, 3.5)
    a.set_xticks(np.arange(-2, 4, 2))
    a.set_ylim(-3.5, 3.5)
    a.set_yticks(np.arange(-2, 4, 2))
    a.set_aspect('equal')
    a.set_xlabel('x')
    if k == 0:
        a.set_ylabel('y')
    if k == 0:
        a.set_title(f'k = {k}, infeasible')
    else:
        a.set_title(f'k = {k}, objective = {val:.1f}')


# problem dimensions and data

n = 6

np.random.seed(1)
idx = np.random.choice(n, 2, replace=False)
w = 1 + np.random.rand(n)
w[idx] = [1, 2]
h = 1 + np.random.rand(n)
h[idx] = [2, 1]
W, H = 5, 6
E = [(0, 1), (2, 3), (0, 2), (1, 3), (2, 4), (3, 4), (5, 0), (5, 1), (5, 3), (5, 4)]


# problem

x = cp.Variable(n)
y = cp.Variable(n)
c = cp.Variable((n, 2))

obj = cp.Minimize(cp.sum([cp.sum_squares(c[i] - c[j]) for i, j in E]))
constr_convex = [c[i] == cp.hstack([x[i] + w[i]/2, y[i] + h[i]/2]) for i in range(n)]
constr_convex += [cp.sum(c, axis=0) == 0]


# solve

K = 5
seeds = [23, 11, 10] if PLOT_ITERATIONS else np.arange(100)
val = np.zeros((len(seeds), K+1))

for idx, seed in enumerate(seeds):
    
    np.random.seed(seed)

    if PLOT_ITERATIONS:
        fig, ax = plt.subplots(1, 4, figsize=(12, 3))

    ck = -2 + 4 * np.random.rand(n, 2)
    xk, yk = ck[:, 0] - w / 2, ck[:, 1] - h / 2
    x.value = xk
    y.value = yk
    c.value = ck
    if PLOT_ITERATIONS:
        draw(ax, 0, xk, yk, ck, obj.value)
    val[idx, 0] = obj.value

    for k in range(K):
        constr_linearized = []
        for i in range(n-1):
            for j in range(i+1, n):
                argmin = np.argmin(np.array([
                    xk[i] + w[i] - xk[j],
                    xk[j] + w[j] - xk[i],
                    yk[i] + h[i] - yk[j],
                    yk[j] + h[j] - yk[i]
                ]))
                if argmin == 0:
                    constr_linearized += [x[i] + w[i] - x[j] <= 0]
                elif argmin == 1:
                    constr_linearized += [x[j] + w[j] - x[i] <= 0]
                elif argmin == 2:
                    constr_linearized += [y[i] + h[i] - y[j] <= 0]
                else:
                    constr_linearized += [y[j] + h[j] - y[i] <= 0]
        prob_linearized = cp.Problem(obj, constr_convex + constr_linearized)
        prob_linearized.solve(solver=cp.MOSEK)
        xk, yk, ck = x.value, y.value, c.value
        if PLOT_ITERATIONS and k < 3:
            draw(ax, k+1, xk, yk, ck, prob_linearized.value)
    
        val[idx, k+1] = prob_linearized.value
    
    if PLOT_ITERATIONS:
        plt.savefig(os.path.join(img_path, f'floor_planning_seed{seed}.pdf'), bbox_inches='tight')
        plt.show()


# plot histogram

if not PLOT_ITERATIONS:
    
    plt.figure(figsize=(5, 4))
    plt.hist(val[:, -1], bins=20, color='blue', edgecolor='black')
    plt.xlabel('final objective')
    plt.ylabel('number of trials')
    plt.savefig(os.path.join(img_path, 'floor_planning_hist.pdf'), bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(5, 4))
    plt.plot(val[:10].T, linewidth=0.5, color='gray')
    plt.xlabel('k')
    plt.xlim(0, K)
    plt.xticks(np.arange(K+1))
    plt.ylabel('objective')
    plt.savefig(os.path.join(img_path, 'floor_planning_traj.pdf'), bbox_inches='tight')
    plt.show()
    
    # compare to lower bound

    best = min(val[:, -1])
    lb = sum(
        min((w[i] + w[j])/2, (h[i] + h[j])/2)**2 for i, j in E
    )
    print(f'best root sum of distances: {np.sqrt(best):.2f}, lower bound: {np.sqrt(lb):.2f}')
