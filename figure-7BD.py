import svg
import numpy as np
import shapely.geometry
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

svg_filename = "data/galago.svg"
dat_filename = "data/galago.csv"

#npy_filename = "output/galago-patch-stipple-1000.npy"
#npy_filename = "output/galago-patch-stipple-5000.npy"
#npy_filename = "output/galago-patch-stipple-10000.npy"
npy_filename = "output/galago-patch-stipple-25000.npy"
#npy_filename = "output/galago-patch-stipple-50000.npy"
png_filename = "figures/figure-7B.png"
pdf_filename = "figures/figure-7B.pdf"
letter = "B"

#npy_filename = "output/galago-inter-stipple-1000.npy"
#npy_filename = "output/galago-inter-stipple-5000.npy"
#npy_filename = "output/galago-inter-stipple-10000.npy"
#npy_filename = "output/galago-inter-stipple-25000.npy"
#npy_filename = "output/galago-inter-stipple-50000.npy"
#png_filename = "figures/figure-7D.png"
#pdf_filename = "figures/figure-7D.pdf"
#letter = "D"

dpi = 100
size = 1000
border = 20

# Collect data
data = np.genfromtxt(dat_filename, names=True, delimiter=';')
density = data["Neu_Dens_SA"]
density /= density.max()
area = data["Surf_area_mm2"]

points = np.load(npy_filename)
points[:,1] = size - points[:,1]
facecolors = np.zeros((len(points), 4))

paths = []
polygons = []
pathnames = []


# Generate path names
n_paths = len(density)
for i in range(0, n_paths):
    pathnames.append("%s" % (1+i))

# Iterating over all paths to get extents
xmin, xmax = 1e10, -1e10
ymin, ymax = 1e10, -1e10
for i,pathname in enumerate(pathnames):
    path = svg.path(svg_filename, pathname)
    verts, codes = path.vertices, path.codes
    paths.append(Path(verts, codes))
    
    xmin = min(xmin, verts[:,0].min())
    xmax = max(xmax, verts[:,0].max())
    ymin = min(ymin, verts[:,1].min())
    ymax = max(ymax, verts[:,1].max())
scale = (size-2*border) / max(abs(xmax-xmin), abs(ymax-ymin))
x_extent = abs(xmax-xmin)*scale
y_extent = abs(ymax-ymin)*scale
x_offset = border # (size - x_extent)//2
y_offset = border # (size - y_extent)//2


fig = plt.figure(figsize=(size/dpi, size/dpi), dpi=dpi)
ax = fig.add_axes([0,0,1,1], aspect=1, frameon=False)

# Resize patches & display them
for i,path in enumerate(paths):
    verts, codes = path.vertices, path.codes
    verts[:,0] = x_offset + (verts[:,0] - xmin) * scale
    verts[:,1] = y_offset + (verts[:,1] - ymin) * scale
    patch = PathPatch(path, alpha=1.0, zorder=-10,
                      linewidth=0.5, linestyle="solid",
                      facecolor="none", edgecolor="black")
    ax.add_patch(patch)
    polygon = shapely.geometry.Polygon(verts)
    polygons.append(polygon)

# Compute density
facecolors[...] = 0,0,0,0.25
D = np.zeros(n_paths)
for i, point in enumerate(points):
    for j, polygon in enumerate(polygons):
        if polygon.contains(shapely.geometry.Point(point)):
            D[j] += 1
            facecolors[i] = 0,0,0,1
            break
print("N =",D.sum())
for j, polygon in enumerate(polygons):
    D[j] /= polygon.area
print("Mean error: %.3f +/- %.3f" % (np.mean(abs(D/D.max()-density)),
                                     np.std(abs(D/D.max()-density))))

# Display points
plt.scatter(points[:,0], points[:,1], s=5,
            edgecolor="none", facecolor=facecolors)

ax.set_xlim(0, size)
ax.set_xticks([])
ax.set_ylim(0, size)
ax.set_yticks([])

ax.text(0.01, 0.99, letter, fontsize=48, weight=700, transform=ax.transAxes,
        ha = "left", va="top")
plt.savefig(png_filename)
plt.savefig(pdf_filename)
plt.show()
