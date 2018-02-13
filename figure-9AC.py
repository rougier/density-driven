import svg
import numpy as np
import shapely.geometry
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist


svg_filename = "data/galago.svg"
dat_filename = "data/galago.csv"
png_filename_P  = "output/galago-patch.png"
png_filename_PA = "figures/figure-9A.png"
pdf_filename_PA = "figures/figure-9A.pdf"
png_filename_I  = "output/galago-inter.png"
png_filename_IC = "figures/figure-9C.png"
pdf_filename_IC = "figures/figure-9C.pdf"

dpi = 100
size = 1000
border = 20

paths = []
points = []
centroids = []

# Collect data
data = np.genfromtxt(dat_filename, names=True, delimiter=';')
density = data["Neu_Dens_SA"]
density /= density.max()

# Generate path names
n_paths = len(density)
pathnames = []
for i in range(0, n_paths):
    pathnames.append("%s" % (1+i))

# Iterating over all paths to get extents
xmin, xmax = 1e10, -1e10
ymin, ymax = 1e10, -1e10
for i,pathname in enumerate(pathnames):
    path = svg.path(svg_filename, pathname)
    verts = path.vertices
    xmin = min(xmin, verts[:,0].min())
    xmax = max(xmax, verts[:,0].max())
    ymin = min(ymin, verts[:,1].min())
    ymax = max(ymax, verts[:,1].max())
scale = (size-2*border) / max(abs(xmax-xmin), abs(ymax-ymin))
x_extent = abs(xmax-xmin)*scale
y_extent = abs(ymax-ymin)*scale
x_offset = (size - x_extent)//2
y_offset = (size - y_extent)//2

# Resize patches
for i,pathname in enumerate(pathnames):
    path = svg.path(svg_filename, pathname)
    verts, codes = path.vertices, path.codes
    verts[:,0] = x_offset + (verts[:,0] - xmin) * scale
    verts[:,1] = y_offset + (verts[:,1] - ymin) * scale

    paths.append(Path(verts, codes))
    points.extend(verts[:-1].tolist())
    centroids.append(shapely.geometry.Polygon(verts).centroid.coords[0])

# Vectorize data
points = np.array(points)
centroids = np.array(centroids)

# Get new extents
xmin = points[:,0].min()
xmax = points[:,0].max()
xrange = abs(xmax-xmin)
ymin = points[:,1].min()
ymax = points[:,1].max()
yrange = abs(ymax-ymin)


# Convex hull
hull = shapely.geometry.MultiPoint(points).convex_hull
H = np.array(hull.exterior.coords)
n = 50
X,Y = H[:,0], H[:,1]
XD = np.diff(X)
YD = np.diff(Y)
D = np.sqrt(XD**2+YD**2)
U = np.cumsum(D)
U = np.hstack([[0],U])
T = np.linspace(0, U.max(), n)
XN = np.interp(T, U, X)
YN = np.interp(T, U, Y)
H = np.dstack((XN, YN)).squeeze()

# Interpolate density values over convex hull
HD = density[cdist(H,centroids).argmin(axis=-1)]
centroids = np.append(centroids, H, axis=0)
density = np.append(density, HD, axis=0)

# Interpolation
n = 512
X = np.linspace(xmin, xmax, n)
Y = np.linspace(ymin, ymax, n)
X,Y = np.meshgrid(X,Y)
XY = np.dstack((X.ravel(), Y.ravel())).squeeze()
Z = griddata(centroids, density, XY, method="cubic", fill_value=0)
Z = Z.reshape(n,n)



# Patchy density map
fig = plt.figure(figsize=(size/dpi, size/dpi), dpi=dpi)
ax = fig.add_axes([0,0,1,1], aspect=1, frameon=False)
for i,path in enumerate(paths):
    d = 1.0-density[i]
    color = (d,d,d)
    patch = PathPatch(path, alpha=1.0,
                      linewidth = 2.0, linestyle="solid",
                      facecolor=color, edgecolor=color)
    ax.add_patch(patch)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0, size)
ax.set_ylim(0, size)

plt.savefig(png_filename_P, dpi=dpi)
ax.text(0.01, 0.99, "A", fontsize=48, weight=700, transform=ax.transAxes,
        ha = "left", va="top")
plt.savefig(png_filename_PA, dpi=dpi)
plt.savefig(pdf_filename_PA)
plt.show()


# Interpolated density map
fig = plt.figure(figsize=(size/dpi, size/dpi), dpi=dpi)
ax = fig.add_axes([0,0,1,1], aspect=1, frameon=False)
plt.imshow(Z, extent=[xmin,xmax,ymin,ymax],
           origin="lower", cmap=plt.cm.gray_r)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0, size)
ax.set_ylim(0, size)

plt.savefig(png_filename_I, dpi=dpi)
ax.text(0.01, 0.99, "C", fontsize=48, weight=700, transform=ax.transAxes,
        ha = "left", va="top")
plt.savefig(png_filename_IC, dpi=dpi)
plt.savefig(pdf_filename_IC)
plt.show()
