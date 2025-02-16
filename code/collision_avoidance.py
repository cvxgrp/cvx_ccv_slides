
import os
import numpy as np
import numpy.linalg as la
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from math import cos, sin


# dimensions and data

np.random.seed(0)

N, T = 10, 63

start = np.zeros((N, 2))
for i in range(N):
    start[i] = [10 * cos((i/N) * 2 * np.pi), 10 * sin((i/N) * 2 * np.pi)]

d = 1


# helper functions

def get_a(Xk, i, j):
    return la.norm(Xk[:, i] - Xk[:, j], axis=1)

def get_b(Xk, i, j):
    return (Xk[:, i] - Xk[:, j]) / get_a(Xk, i, j).reshape(-1, 1)

def get_val(Xk):
    return np.sum(la.norm(Xk[1:] - Xk[:-1], axis=2))


# initialize

Xk = np.zeros((T+1, N, 2))
for t in range(T+1):
    Xk[t] = start * (1 - 2 * t / T)
for t in range(1, T):
    Xk[t] += 1e-5 * np.random.randn(N, 2)


# solve

K = 20

val = np.zeros(K+1)
val[0] = get_val(Xk)

for k in range(K):
    
    X = cp.Variable((T+1, N, 2))
    L = cp.Variable(N)
        
    constr = [X[0] == start, X[-1] == -start]
    for t in range(1, T+1):
        constr += [cp.norm(X[t] - X[t-1], axis=1) <= L/T]
    for i in range(N-1):
        for j in range(i+1, N):
            a = get_a(Xk, i, j)
            b = get_b(Xk, i, j)
            constr += [a + cp.sum(cp.multiply(b, (X[:, i] - X[:, j] - Xk[:, i] + Xk[:, j])), axis=1) >= d]
    
    prob = cp.Problem(cp.Minimize(cp.sum(L)), constr)
    prob.solve()
    
    Xk = X.value
    val[k+1] = get_val(Xk)
    

# draw

colors = plt.cm.jet(np.linspace(0, 1, N))
colors = colors[np.random.permutation(N)]
colors_dark = 0.5 * colors

def draw(Xk, colored=False):
    for i, (x, y) in enumerate(Xk):
        facecolor = colors[i] if colored else 'lightgray'
        edgecolor = colors_dark[i] if colored else 'gray'
        circle = Circle((x, y), d/2, edgecolor=edgecolor, facecolor=facecolor, alpha=1)
        plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.xlim([-12, 12])
    plt.ylim([-12, 12])
    plt.xticks([])
    plt.yticks([])

plt.figure(figsize=(4.2, 4))
for x, y in start:
    plt.annotate('', xy=(x/2, y/2), xytext=(x, y), arrowprops=dict(facecolor='black', arrowstyle='->'), zorder=-1)
draw(start, True)
plt.savefig('img/collision_avoidance_setting.pdf', bbox_inches='tight')
plt.show()

for t in range(T+1):
    plt.figure(figsize=(4.2, 4))
    draw(Xk[t], True)
    plt.savefig(f'img/collision_avoidance_frame_{t}.png', bbox_inches='tight')
    plt.close()
    
gif_duration = 3
frame_duration = int((gif_duration / T) * 1000)
frames = []
for t in range(T+1):
    file = f'img/collision_avoidance_frame_{t}.png'
    if os.path.exists(file):
        frames.append(Image.open(file))
        os.remove(file)

output_path = 'img/collision_avoidance.gif'
frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=frame_duration,
    loop=0
)
print(f"GIF saved at {output_path}")
    
plt.figure(figsize=(8, 4))
for t in range(0, T+1, 9):
    plt.subplot(2, 4, t // 9 + 1)
    for i in range(N):
        #plt.plot(Xk[:t, i, 0], Xk[:t, i, 1], 'gray', linewidth=0.3, zorder=-1)
        # add arrows
        if t < T:
            x, y = Xk[t, i, 0], Xk[t, i, 1]
            xplus, yplus = x + 8 * (Xk[t+1, i, 0] - x), y + 8 * (Xk[t+1, i, 1] - y)
            plt.annotate(
                '', xy=(xplus, yplus), xytext=(x, y),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                zorder=10 if t == 27 else -1
            )
    draw(Xk[t], True)
    plt.title(f'$t={t}$')
plt.tight_layout()
plt.savefig('img/collision_avoidance_frames.pdf', bbox_inches='tight')
plt.show()
