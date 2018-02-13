# Copyright (2017) Nicolas P. Rougier - BSD license
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
matplotlib.rc('xtick.major', size=10)

xmin, xmax = 0, 1024
ymin, ymax = 0, 256

# ./stippler.py data/gradient-1024x256.png --n_iter 50 --n_point 1000 \
#   --save --force --seed 1 --pointsize 1.0 1.0 --figsize 6 --interactive
P1 = np.load("output/gradient-1024x256-stipple-1000.npy")

# ./stippler.py data/gradient-1024x256.png --n_iter 50 --n_point 2500 \
#   --save --force --seed 1 --pointsize 1.0 1.0 --figsize 6 --interactive
P2 = np.load("output/gradient-1024x256-stipple-2500.npy")

# ./stippler.py data/gradient-1024x256.png --n_iter 50 --n_point 5000 \
#   --save --force --seed 1 --pointsize 1.0 1.0 --figsize 6 --interactive
P3 = np.load("output/gradient-1024x256-stipple-5000.npy")

# ./stippler.py data/gradient-1024x256.png --n_iter 50 --n_point 10000 \
#   --save --force --seed 1 --pointsize 1.0 1.0 --figsize 6 --interactive
P4 = np.load("output/gradient-1024x256-stipple-10000.npy")

def plot(ax, P, linewidth = 0.5, size = 25):
    reference = 1000
    ratio = reference/len(P)
    size      = max(ratio * size, 3.0)
    linewidth = max(ratio * linewidth, 0.25)
    ax.scatter(P[:,0], P[:,1], facecolor="white", edgecolor="black",
               s=size, linewidth=linewidth)
    ax.set_ylabel("%d cells" % len(P))
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(np.linspace(xmin,xmax,5, endpoint=True))
    ax.set_xticklabels(["",]*5)

    X = np.linspace(xmin,xmax,5, endpoint=True)
    for x0,x1 in zip(X[:-1], X[1:]):
        n = np.logical_and(P[:,0] >= x0, P[:,0] < x1).sum()
        ratio = 100*n/len(P)
        ax.text((x1+x0)/2, -6, "%.2f%% (n=%d)" % (ratio, n), fontsize=8,
                ha="center", va="top", clip_on=False)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([])

plt.figure(figsize=(10,10))

ax = plt.subplot(4,1,1, aspect=1)
plot(ax, P1)
ax = plt.subplot(4,1,2, aspect=1)
plot(ax, P2)
ax = plt.subplot(4,1,3, aspect=1)
plot(ax, P3)
ax = plt.subplot(4,1,4, aspect=1)
plot(ax, P4)

plt.tight_layout()
plt.savefig("figures/figure-4.pdf")
plt.show()
