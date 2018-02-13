import numpy as np

def rasterize(V):
    """
    Polygon rasterization (scanlines).

    Given an ordered set of vertices V describing a polygon,
    return all the (integer) points inside the polygon.
    See http://alienryderflex.com/polygon_fill/

    Parameters:
    -----------

    V : (n,2) shaped numpy array
        Polygon vertices
    """

    n = len(V)
    X, Y = V[:, 0], V[:, 1]
    ymin = int(np.ceil(Y.min()))
    ymax = int(np.floor(Y.max()))
    #ymin = int(np.round(Y.min()))
    #ymax = int(np.round(Y.max()))
    P = []
    for y in range(ymin, ymax+1):
        segments = []
        for i in range(n):
            index1, index2 = (i-1) % n, i
            y1, y2 = Y[index1], Y[index2]
            x1, x2 = X[index1], X[index2]
            if y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = x2, x1
            elif y1 == y2:
                continue
            if (y1 <= y < y2) or (y == ymax and y1 < y <= y2):
                segments.append((y-y1) * (x2-x1) / (y2-y1) + x1)

        segments.sort()
        for i in range(0, (2*(len(segments)//2)), 2):
            x1 = int(np.ceil(segments[i]))
            x2 = int(np.floor(segments[i+1]))
            # x1 = int(np.round(segments[i]))
            # x2 = int(np.round(segments[i+1]))
            P.extend([[x, y] for x in range(x1, x2+1)])
    if not len(P):
        return V
    return np.array(P)


def centroid(vertices):
    A = 0
    Cx = 0
    Cy = 0
    for i in range(len(vertices)-1):
        s = vertices[i, 0] * vertices[i+1, 1] - vertices[i+1, 0] * vertices[i, 1]
        A += s
        Cx += (vertices[i, 0] + vertices[i+1, 0]) * s
        Cy += (vertices[i, 1] + vertices[i+1, 1]) * s
    Cx /= 3*A
    Cy /= 3*A
    return np.array([[Cx, Cy]])

def weighted_centroid(vertices):
    V = rasterize(vertices)
    return V.sum(axis=0)/float(len(V))


def plot_shape(ax, seed, radius, letter):

    np.random.seed(seed)
    n = 12

    T = np.linspace(0, 2*np.pi, n, endpoint=True)
    R = radius * np.random.uniform(0.75, 1.25, n)
    R[-1] = R[0]
    
    V = np.zeros((n,2))
    V[:,0] = R*np.cos(T)
    V[:,1] = R*np.sin(T)
    V[:,0] -= V[:,0].min()
    V[:,1] -= V[:,1].min()

    P = rasterize(V)
    C = centroid(V)
    wC = weighted_centroid(V[:-1])

    Pi = np.round(P).astype(int)
    shape = Pi[:,1].max()+1, Pi[:,0].max()+1
    D = np.zeros(shape)
    D[Pi[:,1], Pi[:,0]] = 1
    ax.imshow(.15*D, extent=[0, D.shape[1], 0, D.shape[0]], origin="lower",
              interpolation="nearest", vmin=0, vmax=1, cmap=plt.cm.gray_r, )

    ax.plot(V[:,0], V[:,1], color="k", linewidth=2)
    ax.scatter(V[:,0], V[:,1], s=25,  edgecolor="k", facecolor="w", zorder=10)
    ax.scatter(C[:,0], C[:,1], s=50, marker="o", linewidth=1,
               edgecolor="k", facecolor="w", alpha=1, zorder=10)
    ax.scatter(wC[0], wC[1], s=100, marker="x", linewidth=2,
               edgecolor="None", facecolor="k", zorder=20)

    ax.set_xlim(-0.5,shape[1]+0.5)
    ax.set_xticks(np.arange(shape[1]+1))
    ax.set_xticklabels([])
    ax.xaxis.set_ticks_position('none')

    ax.set_ylim(-0.5,shape[0]+0.5)
    ax.set_yticks(np.arange(shape[0]+1))
    ax.set_yticklabels([])
    ax.yaxis.set_ticks_position('none') 

    ax.text(.05,.95, letter, fontsize="xx-large", weight="bold",
            ha="center", va="center", transform=ax.transAxes)
    
    ax.grid(color="0.6", linestyle='solid')

    
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(12,6))
ax = plt.subplot(2,3,1, frameon=False)
plot_shape(ax, 1, 3, "A")

ax = plt.subplot(2,3,2, frameon=False)
plot_shape(ax, 1, 5, "B")

ax = plt.subplot(2,3,3, frameon=False)
plot_shape(ax, 1, 10, "C")


ax = plt.subplot(2,1,2)


def difference(seed, radius):
    np.random.seed(seed)
    n = 16
    T = np.linspace(0, 2*np.pi, n, endpoint=True)
    R = radius * np.random.uniform(0.5, 1.5, n)
    R[-1] = R[0]
    V = np.zeros((n,2))
    V[:,0] = R*np.cos(T)
    V[:,1] = R*np.sin(T)
    V[:,0] -= V[:,0].min()
    V[:,1] -= V[:,1].min()
    P = rasterize(V)
    C = centroid(V)
    wC = weighted_centroid(V[:-1])
    return np.sqrt(np.sum((C-wC)**2))


P = np.zeros((100,50))
for i,radius in enumerate(np.linspace(2,50,P.shape[0])):
    for seed in range(P.shape[1]):
        P[i,seed] = difference(seed, radius)
X = np.linspace(2,50,P.shape[0])
D = np.mean(P,axis=1)
S = np.std(P,axis=1)
ax.plot(X,D)
ax.fill_between(X, D+S, D-S, alpha=.15)


ax.scatter( [X[6],X[17],X[37]], [D[6],D[17],D[37]], s=25,
             facecolor="w", edgecolor="k", zorder=30)
dy = 0.01
ax.text(X[6], D[6]-dy, "A", fontsize="large",
        ha="center", va="top", transform=ax.transData)
ax.text(X[17], D[17]-dy, "B", fontsize="large", 
        ha="center", va="top", transform=ax.transData)
ax.text(X[37], D[37]-dy, "C", fontsize="large", 
        ha="center", va="top", transform=ax.transData)


ax.set_xticks([10,20,30,40,50])
ax.set_xticklabels(["10×10","20×20","30×30","40×40","50×50"])

ax.set_xlabel("Size (pixels²)")
ax.set_ylabel("Precision (pixels)")


plt.tight_layout()
plt.savefig("figures/figure-5.pdf")

plt.show()



