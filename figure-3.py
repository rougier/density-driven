# Copyright (2017) Nicolas P. Rougier - BSD license
import os
import tqdm
import numpy as np
import scipy.spatial.distance
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Polygon
from matplotlib.collections import PatchCollection
import voronoi
from stippler import normalize, initialization


# Parameters
# ----------
n = 100
xmin, xmax = 0, 1024
ymin, ymax = 0, 1024
np.random.seed(123)
force = False

# Uniform density
density = np.ones((xmax-xmin,xmax-xmin))
density_P = density.cumsum(axis=1)
density_Q = density_P.cumsum(axis=1)

# Centroidal Voronoi Tesselation (CVT)
# ------------------------------------
if not os.path.exists("output/CVT-initial.npy") or force:
    points = np.zeros((n,2))
    points[:,0] = np.random.uniform(xmin, xmax, n)
    points[:,1] = np.random.uniform(ymin, ymax, n)
    np.save("output/CVT-initial.npy", points)
    for i in tqdm.trange(50):
        regions, points = voronoi.centroids(points, density, density_P, density_Q)
    np.save("output/CVT-final.npy", points)


# Display
# -------
def plot(ax, points, letter):

    patches = []
    regions, vertices = voronoi.voronoi_finite_polygons_2d(points)
    for region in regions:
        patches.append(Polygon(vertices[region]))
    collection = PatchCollection(patches, linewidth=0.75,
                                 facecolor="white", edgecolor="black", )
    ax.add_collection(collection)
    ax.scatter(points[:,0], points[:,1], s=15,
               facecolor="red", edgecolor="none")
    regions, points = voronoi.centroids(points, density, density_P, density_Q)
    ax.scatter(points[:,0], points[:,1], s=50,
               facecolor="none", edgecolor="black", linewidth=.75)
    ax.text(24, ymax-24, letter, color="black", weight="bold", va="top", fontsize=24)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks([])
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([])

    
plt.figure(figsize=(10,5))

ax = plt.subplot(1, 2, 1, aspect=1)
points = np.load("output/CVT-initial.npy")
plot(ax, points, "A")

ax = plt.subplot(1, 2, 2, aspect=1)
points = np.load("output/CVT-final.npy")
plot(ax, points, "B")

plt.tight_layout()
plt.savefig("figures/figure-3.pdf")
plt.show()
