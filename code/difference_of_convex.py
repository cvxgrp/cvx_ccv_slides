import os
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


img_path = 'img'
if os.path.basename(os.getcwd()) == 'code':
    img_path = os.path.join('..', img_path)


# functions

f1 = lambda x: x**4 + 0.2*x
g1 = lambda x: x**2
h1 = lambda x: f1(x) - g1(x)

f2 = lambda x: np.maximum(0, x**3)
g2 = lambda x: np.maximum(0, -x**3)
h2 = lambda x: f2(x) - g2(x)


# plot f, g, and h

x = np.linspace(-1, 1, 100)

plt.plot(x, f1(x), '--', color='black')
plt.plot(x, g1(x), '--', color=[1, 0, 1])
plt.plot(x, h1(x), color='blue')
plt.xticks(np.arange(-1, 2))
plt.yticks(np.arange(0, 2))
plt.xlabel('x')
plt.savefig(os.path.join(img_path, 'DC1.pdf'))
plt.show()

plt.plot(x, h2(x), color='blue')
plt.plot(x, f2(x), '--', color='black')
plt.plot(x, g2(x), '--', color=[1, 0, 1])
plt.xticks(np.arange(-1, 2))
plt.yticks(np.arange(-1, 2))
plt.xlabel('x')
plt.savefig(os.path.join(img_path, 'DC2.pdf'))
plt.show()


# plot \hat h

def get_majorizer1(xk):
    def m(x):
        return f1(x) - (g1(xk) + 2 * xk * (x - xk))
    return m

def get_majorizer2(xk):
    def m(x):
        return f2(x) - (g2(xk) + min(-3 * xk**2, 0) * (x - xk))
    return m

plt.plot(x, get_majorizer1(0.3)(x), color='gray', linewidth=0.7)
plt.plot(x, h1(x), color='blue')
plt.xticks(np.arange(-1, 2))
plt.yticks(np.arange(0, 2))
plt.xlabel('x')
plt.savefig(os.path.join(img_path, 'DC1_majorized.pdf'))
plt.show()

plt.plot(x, get_majorizer2(-0.5)(x), color='gray', linewidth=0.7)
plt.plot(x, h2(x), color='blue')
plt.xticks(np.arange(-1, 2))
plt.yticks(np.arange(-1, 3))
plt.xlabel('x')
plt.savefig(os.path.join(img_path, 'DC1_majorized.pdf'))
plt.show()


# four iterions of CCP, for two different initializations

x_cp = cp.Variable()

for xk, name in zip([-0.2, 0.2], ['left', 'right']):

    plt.figure(figsize=(7, 4))

    for k in range(4):
            
        hhat = get_majorizer1(xk)
        
        plt.subplot(2, 2, k+1)
        
        plt.plot(x, h1(x), color='blue')
        plt.plot(x, hhat(x), '-', color='gray', linewidth=0.7)
        plt.scatter(xk, h1(xk), color='gray', marker='s', zorder=10)
        
        prob = cp.Problem(cp.Minimize(hhat(x_cp)))
        prob.solve()
        xk = x_cp.value
        
        plt.scatter(xk, h1(xk), color='black', zorder=10)
            
        plt.xticks(np.arange(-1, 2))
        plt.yticks(np.arange(-0.4, 0.2, 0.3))
        plt.ylim(-0.45, 0.25)
        
        if k>1:
            plt.xlabel('x')
            
        plt.title(f'$k={k}$')

    plt.tight_layout()
    plt.savefig(os.path.join(img_path, f'DC1_iterations_{name}.pdf'))
    plt.show()
