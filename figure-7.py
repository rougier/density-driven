# A graphical, scalable and intuitive method for the placement and connections
# of biological cells - Copyright (2017) Nicolas P. Rougier - BSD license
import tqdm
import numpy as np
from scipy import interpolate
import scipy.spatial.distance
import voronoi
from stippler import normalize, initialization
from voronoi import voronoi_finite_polygons_2d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path

def polygon_area(P):
    lines = np.hstack([P,np.roll(P,-1,axis=0)])
    return 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))


def blob(center, radius):
    n = 10
    noise = 0.4
    T = np.linspace(0, 2*np.pi, n, endpoint=False)
    R = np.random.uniform(1-noise/2, 1+noise/25, n) * radius
    X, Y = center[0]+R*np.cos(T),  center[1]+R*np.sin(T)
    X = np.r_[X,X[0]]
    Y = np.r_[Y,Y[0]]
    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    TCK, U = interpolate.splprep([X, Y], s=0, per=True)
    # evaluate the spline fits for 1000 evenly spaced distance values
    Xi, Yi = interpolate.splev(np.linspace(0, 1, 1000), TCK)
    
    verts = np.dstack([Xi,Yi]).reshape(len(Xi),2)
    codes = [Path.MOVETO,] + [Path.LINETO,]*(len(verts)-2) + [Path.LINETO,]
    path = Path(verts, codes)
    # return Xi, Yi
    return X, Y, path

    
# Parameters
# ----------
np.random.seed(1)

n = 1024
xmin, xmax = 0, n
ymin, ymax = 0, n
n_cones = 25
n_rods = 2500
n_iter_cones = 15
kn_iter_rods = 30
cones_radius = 30
force = 0


# Compute cones locations
# -----------------------
x0, y0 = (xmin+xmax)/2, (ymin+ymax)/2
X, Y = np.meshgrid(np.linspace(xmin, xmax, n, endpoint=False),
                   np.linspace(ymin, ymax, n, endpoint=False))
C = np.sqrt((X-x0)*(X-x0)+(Y-y0)*(Y-y0))
C = normalize(C)
density = 1-np.power(C,0.5)
density_P = density.cumsum(axis=1)
density_Q = density_P.cumsum(axis=1)
points = initialization(n_cones, density)
cones_density = density

if force:
    print("Generating cones locations (n=%d)" % len(points))
    for i in tqdm.trange(n_iter_cones):
        regions, points = voronoi.centroids(points, density, density_P, density_Q)
    cones = points
    np.save("output/cones.npy", cones)
    np.save("output/cones_density.npy", cones_density)
    cones_radii = cones_radius * np.random.uniform(0.9,1.1,len(cones))
    np.save("output/cones_radii.npy", cones_radii)
else:
    cones = np.load("output/cones.npy")
    cones_radii = np.load("output/cones_radii.npy")
    cones_density = np.load("output/cones_density.npy")
    print("Loading cones locations and radii (n=%d)" % len(cones))


# Compute rods locations
# ----------------------
if force:
    density = np.zeros((n,n))
    density[:] = np.linspace(0.00,0.5,n)
    
    for i,(x,y) in enumerate(points):
        C = np.sqrt((X-x)*(X-x)+(Y-y)*(Y-y))
        density[C < cones_radii[i]] = 1
        
    density = 1-normalize(density)
    density_P = density.cumsum(axis=1)
    density_Q = density_P.cumsum(axis=1)
    rods_density = density
    points = initialization(n_rods, density)
    print("Generating rods locations (n=%d)" % len(points))
    for i in tqdm.trange(n_iter_rods):
        regions, points = voronoi.centroids(points, density, density_P, density_Q)
    rods = points
    np.save("output/rods.npy", rods)
    np.save("output/rods_density.npy", rods_density)
else:
    rods = np.load("output/rods.npy")
    rods_density = np.load("output/rods_density.npy")
    print("Loading rods locations (n=%d)" % len(rods))


# Display
# -------
plt.figure(figsize=(9,6))

ax = plt.subplot2grid((2,3), (0, 0), aspect=1)

ax.imshow(1-cones_density, extent=[xmin, xmax, ymin, ymax],
           cmap=plt.get_cmap("gray"), origin="lower")
ax.text(24, ymax-24, "A", color="black", weight="bold", va="top", fontsize=16)
ax.text(12, 12, "Bitmap (1024x1024)", color="black", va="bottom", fontsize=8)
ax.set_xticks([])
ax.set_yticks([])


ax = plt.subplot2grid((2,3), (1, 0), aspect=1)
ax.imshow(1-rods_density, extent=[xmin, xmax, ymin, ymax],
           cmap=plt.get_cmap("gray"), origin="lower")
ax.text(24, ymax-24, "B", color="white", weight="bold", va="top", fontsize=16)
ax.text(12, 12, "Bitmap (1024x1024)", color="white", va="bottom", fontsize=8)

ax.set_xticks([])
ax.set_yticks([])


ax = plt.subplot2grid((2,3), (0, 1), colspan=2, rowspan=2, aspect=1)
facecolor = np.zeros((len(cones),4))
facecolor[:,1] = facecolor[:,2] = np.random.uniform(0.50, 0.65, len(cones))
facecolor[:,0] = facecolor[:,3] = 1


for P in cones:
   X, Y, path = blob(P, 1.15*cones_radius)
   patch = PathPatch(path, edgecolor="black", facecolor="white")
   ax.add_patch(patch)

points = np.append(cones, rods).reshape(len(cones)+len(rods), 2)
patches = []
regions, vertices = voronoi_finite_polygons_2d(points)

facecolor = np.zeros((len(regions),4))

for i,region in enumerate(regions):
    patches.append(Polygon(vertices[region]))
    a = np.random.uniform(0.75,1.00)
    facecolor[i] = a,a,a,1

collection = PatchCollection(patches, linewidth=0.25,
                             facecolor=facecolor, edgecolor="black")
ax.add_collection(collection)

ax.set_xlim(xmin, xmax)
ax.set_xticks([])
ax.set_ylim(ymin, ymax)
ax.set_yticks([])
ax.text(24, ymax-24, "C", color="black", weight="bold",
        va="top", fontsize=24, zorder=100)

plt.tight_layout()
plt.savefig("figures/figure-7.pdf")
plt.savefig("figures/figure-7.png")
plt.show()
