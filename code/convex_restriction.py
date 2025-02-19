import os
import cvxpy as cp
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


img_path = 'img'
if os.path.basename(os.getcwd()) == 'code':
    img_path = os.path.join('..', img_path)


markersize=30

def draw_restriction(x1, xk):
    
    circle = Circle((0, 0), 1, facecolor='black')
    plt.gca().add_patch(circle)

    x2 = -(xk[0] / xk[1]) * x1 + la.norm(xk) / xk[1]
    plt.fill_between(x1, x2, 10, where=(x2 <= 10), color=0.95*np.ones(3))
    plt.scatter(xk[0], xk[1], markersize, color='darkgray', marker='s')

    # Adjust limits and aspect ratio
    plt.xlim(min(x1), max(x1))
    plt.xticks(np.arange(min(x1), max(x1) + 1, 2))
    plt.ylim(min(x1), max(x1))
    plt.yticks(np.arange(min(x1), max(x1) + 1, 2))


x1 = np.linspace(-2, 4, 100)

plt.figure(figsize=(4, 4))
draw_restriction(x1, np.array([3, 1]))
plt.savefig(os.path.join(img_path, 'convex_restriction_31.pdf'), bbox_inches='tight')
plt.show()

plt.figure(figsize=(4, 4))
draw_restriction(x1, np.array([1, 2]))
plt.savefig(os.path.join(img_path, 'convex_restriction_12.pdf'), bbox_inches='tight')
plt.show()

# 4 iterations of CCP

c = np.array([-0.4, 0.6])

x = cp.Variable(2)
xk_normalized = cp.Parameter(2)
obj = cp.Minimize(cp.norm(x - c))
constr = [1 - xk_normalized @ x <= 0]
prob = cp.Problem(obj, constr)

plt.figure(figsize=(6, 6.2))

x1 = np.linspace(-3, 3, 100)
xk = np.array([2.5, 1])

K = 6
val = np.zeros(K + 1)
val[0] = la.norm(xk - c)

for k in range(K):
    
    xk_normalized.value = xk / la.norm(xk)
    prob.solve()
    
    if k < 4:
        plt.subplot(2, 2, k+1)
        draw_restriction(x1, xk)
        plt.scatter(x.value[0], x.value[1], markersize, color='blue')
        plt.scatter(c[0], c[1], markersize, color='white', marker='x')
        plt.title(f'$k={k}$')
    
    xk = x.value
    val[k+1] = la.norm(xk - c)
    
plt.tight_layout()
plt.savefig(os.path.join(img_path, 'convex_restriction_iterations.pdf'), bbox_inches='tight')
plt.show()

np.random.seed(5)

n_other = 10
val_other = np.zeros((n_other, K + 1))

for i in range(n_other):

    xk = np.random.rand(2) * 4 - 2
    xk = 2 * xk / min(1, la.norm(xk))
    val_other[i, 0] = la.norm(xk - c)

    for k in range(K):
        
        xk_normalized.value = xk / la.norm(xk)
        prob.solve()
        xk = x.value
        val_other[i, k+1] = la.norm(xk - c)

plt.figure(figsize=(6, 4))
plt.plot(val_other.T, color=[0.8, 0.8, 1], linewidth=1)
plt.plot([0, K], (1-la.norm(c)) * np.ones(2), '--', color='black')
plt.plot(val, color='blue')
plt.xticks(np.arange(K+1))
plt.xlabel('k')
plt.ylabel('objective')
plt.savefig(os.path.join(img_path, 'convex_restriction_iterations_value.pdf'), bbox_inches='tight')
plt.show()
