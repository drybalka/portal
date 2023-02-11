import numpy as np
import matplotlib.pyplot as plt

ny = 40
nx = 30
width = 10

ph_left = 8
ph_top = 29
ph_bottom = 30
pv_top = 10
pv_left = 17
pv_right = 18


def top(u, y, x):
    if y == 0:
        return u[y, x] + 1
    elif y == ph_bottom and x in range(ph_left, ph_left + width):
        return u[pv_top + x - ph_left, pv_right]
    else:
        return u[y - 1, x]


def bottom(u, y, x):
    if y == ny - 1:
        return u[y, x] - 1
    elif y == ph_top and x in range(ph_left, ph_left + width):
        return u[pv_top + x - ph_left, pv_left]
    else:
        return u[y + 1, x]


def left(u, y, x):
    if x == pv_right and y in range(pv_top, pv_top + width):
        return u[ph_bottom, ph_left + y - pv_top]
    else:
        return u[y, max(x - 1, 0)]


def right(u, y, x):
    if x == pv_left and y in range(pv_top, pv_top + width):
        return u[ph_top, ph_left + y - pv_top]
    return u[y, min(x + 1, nx - 1)]


def laplacian(u, y, x):
    return top(u, y, x) + bottom(u, y, x) + left(u, y, x) + right(
        u, y, x) - 4 * u[y, x]


def gradY(u):
    gradU = np.empty((ny, nx))
    for y in range(0, ny):
        for x in range(0, nx):
            gradU[y, x] = (bottom(u, y, x) - top(u, y, x)) / 2
    return gradU


def gradX(u):
    gradU = np.empty((ny, nx))
    for y in range(0, ny):
        for x in range(0, nx):
            gradU[y, x] = (left(u, y, x) - right(u, y, x)) / 2
    return gradU


def step(u):
    newU = np.empty((ny, nx))
    for y in range(0, ny):
        for x in range(0, nx):
            newU[y, x] = u[y, x] + 0.1 * laplacian(u, y, x)
    return newU


def initialize():
    u = np.empty((ny, nx))
    for y in range(0, ny):
        for x in range(0, nx):
            u[y, x] = ny / 2 - y
    return u


u = initialize()

for i in range(0, 5000):
    u = step(u)

dx = gradX(u)
dy = gradY(u)

fig, ax = plt.subplots(figsize=(6, 8))
ax.imshow(u)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.quiver(dx, dy)
plt.plot([ph_left, ph_left + width - 1], [(ph_bottom + ph_top) / 2,
                                      (ph_bottom + ph_top) / 2],
         color='orange',
         linewidth=2)
plt.plot([(pv_left + pv_right) / 2, (pv_left + pv_right) / 2],
         [pv_top, pv_top + width - 1],
         color='blue',
         linewidth=2)
plt.savefig('portal_orthogonal.png')
