import numpy as np
import matplotlib.pyplot as plt

ny = 30
nx = 30

p1_top = 7
p1_bottom = p1_top + 1
p2_bottom = ny - p1_bottom
p2_top = p2_bottom - 1

p1_left = 11
p1_right = nx - p1_left
p2_left = p1_left
p2_right = p1_right


def top(u, y, x):
    if y == 0:
        return u[y, x] + 1
    elif y == p1_bottom and x in range(p1_left, p1_right):
        return u[p2_top, x]
    elif y == p2_bottom and x in range(p1_left, p1_right):
        return u[p1_top, x]
    else:
        return u[y - 1, x]


def bottom(u, y, x):
    if y == ny - 1:
        return u[y, x] - 1
    elif y == p1_top and x in range(p1_left, p1_right):
        return u[p2_bottom, x]
    elif y == p2_top and x in range(p1_left, p1_right):
        return u[p1_bottom, x]
    else:
        return u[y + 1, x]


def left(u, y, x):
    return u[y, max(x - 1, 0)]


def right(u, y, x):
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

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(u)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.quiver(dx, dy)
plt.plot([p1_left, p1_right - 1], [(p1_top + p1_bottom) / 2,
                                   (p1_top + p1_bottom) / 2],
         color='orange',
         linewidth=2)
plt.plot([p2_left, p2_right - 1], [(p2_top + p2_bottom) / 2,
                                   (p2_top + p2_bottom) / 2], color='blue', linewidth=2)
plt.savefig('portal.png')
