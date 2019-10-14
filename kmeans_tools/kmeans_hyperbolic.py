import math
import cmath
import torch
import numpy as np
import tqdm
import random

from function_tools import poincare_alg as pa
from function_tools import poincare_function as pf

import pytorch_categorical
class PoincareKMeans(object):
    def __init__(self, n_clusters, min_cluster_size=5, verbose=False, init_method="random"):
        self._n_c = n_clusters
        self._distance = pf.distance
        self.centroids = None
        self._mec = min_cluster_size
        self._init_method = init_method

    def _maximisation(self, x, indexes):
        centroids = x.new(self._n_c, x.size(-1))
        for i in range(self._n_c):
            lx = x[indexes == i]
            if(lx.shape[0] <= self._mec):
                lx = x[random.randint(0,len(x)-1)].unsqueeze(0)
            centroids[i] = pa.barycenter(lx, normed=True)
        return centroids
    
    def _expectation(self, centroids, x):
        N, K, D = x.shape[0], self.centroids.shape[0], x.shape[1]
        centroids = centroids.unsqueeze(0).expand(N, K, D)
        x = x.unsqueeze(1).expand(N, K, D)
        dst = self._distance(centroids, x)
        value, indexes = dst.min(-1)
        return indexes

    def _init_random(self, X):
        self.centroids_index = (torch.rand(self._n_c, device=X.device) * len(X)).long()
        self.centroids = X[self.centroids_index]

    def __init_kmeansPP(self, X):
        distribution = torch.ones(len(X))/len(X)
        frequency = pytorch_categorical.Categorical(distribution)
        centroids_index = []
        N, D = X.shape
        while(len(centroids_index)!=self._n_c):

            f = frequency.sample(sample_shape=(1,1)).item()
            if(f not in centroids_index):
                centroids_index.append(f)
                centroids = X[centroids_index]
                x = X.unsqueeze(1).expand(N, len(centroids_index), D)
                dst = self._distance(centroids, x)
                value, indexes = dst.min(-1)
                vs = value**2
                distribution = vs/(vs.sum())
                frequency = pytorch_categorical.Categorical(distribution)
        self.centroids_index = torch.tensor(centroids_index, device=X.device).long()
        self.centroids = X[self.centroids_index]

    def fit(self, X, max_iter=500):
        with torch.no_grad():
            if(self._mec < 0):
                self._mec = len(X)/(self._n_c**2)
            if(self.centroids is None):
                if(self._init_method == "kmeans++"):
                    self.__init_kmeansPP(X)
                else:
                    self._init_random(X)
            for iteration in range(max_iter):
                if(iteration >= 1):
                    old_indexes = self.indexes
                self.indexes = self._expectation(self.centroids, X)
                self.centroids = self._maximisation(X, self.indexes)
                if(iteration >= 1):
                    if((old_indexes == self.indexes).float().mean() == 1):
                        self.cluster_centers_  =  self.centroids
                        return self.centroids
            self.cluster_centers_  =  self.centroids
            return self.centroids

    def predict(self, X):
        return self._expectation(self.centroids, X)

    def getStd(self, x):
        N, K, D = x.shape[0], self.centroids.shape[0], x.shape[1]
        centroids = self.centroids.unsqueeze(0).expand(N, K, D)
        x = x.unsqueeze(1).expand(N, K, D)
        dst = self._distance(centroids, x)**2
        value, indexes = dst.min(-1)
        stds = []
        for i in range(self._n_c):
            stds.append(value[indexes==i].sum())
        stds = torch.Tensor(stds)
        return stds

# def test():
#     import torch
#     import matplotlib.pyplot as plt
#     from matplotlib.patches import Circle
#     import numpy as np
#     from itertools import product, combinations
#     from mpl_toolkits.mplot3d import Axes3D

#     x1 = torch.randn(500, 2)*0.10 +(torch.rand(1, 2).expand(500, 2) -0.5) * 3
#     x2 = torch.randn(500, 2)*0.10 +(torch.rand(1, 2).expand(500, 2) -0.5) * 3
#     x3 = torch.randn(500, 2)*0.10 +(torch.rand(1, 2).expand(500, 2) -0.5) * 3
#     X = torch.cat((x1,x2,x3), 0)
#     X_b = torch.cat((x1.unsqueeze(0),x2.unsqueeze(0),x3.unsqueeze(0)), 0)
#     xn  = X.norm(2,-1)

#     X[xn>1] /= ((xn[xn>1]).unsqueeze(-1).expand((xn[xn>1]).shape[0], 2) +1e-3)
#     X_b = torch.cat((X[0:500].unsqueeze(0),X[500:1000].unsqueeze(0),X[1000:].unsqueeze(0)), 0)
#     km = PoincareKMeans(3, min_cluster_size=10)
#     import time
#     start_time = time.time()
#     print("start fitting")
#     mu = km.fit(X.cuda())
#     #    mu = km.fit(X)
#     end_time = time.time()
#     print("end fitting")
#     # took ~31 seconds for 150000 data on gpu 1070 gtx 50 epochs
#     # took ~125 seconds on CPU

#     print("Time to fit -> "+str(end_time-start_time))
#     ax = plt.subplot()
#     p = Circle((0, 0), 1, edgecolor='b', lw=1, facecolor='none')
#     ax.add_patch(p)
#     plt.scatter(X[:100,0].numpy(), X[:100,1].numpy())
#     plt.scatter(X[500:600,0].numpy(), X[500:600,1].numpy())
#     plt.scatter(X[1000:1100,0].numpy(), X[1000:1100,1].numpy())
#     print(mu)
#     print(mu.shape)
#     plt.scatter(mu[:,0].cpu().numpy(),mu[:,1].cpu().numpy(), label="Poincare barycenter",
#                 marker="s", c="red", s=100.)
#     plt.scatter(X_b.mean(1)[:,0], X_b.mean(1)[:,1], label="Euclidean barycenter by real clusters",
#                 marker="s", c="green", s=100.)
#     plt.legend()
#     plt.show()
#     print("3D")

#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     #ax.set_aspect("equal")
#     # draw sphere
#     u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#     x = np.cos(u)*np.sin(v)
#     y = np.sin(u)*np.sin(v)
#     z = np.cos(v)
#     ax.plot_wireframe(x, y, z, color="r")
#     x1 = torch.randn(100, 3)*0.2 +(torch.rand(1, 3).expand(100, 3) -0.5) * 3
#     x2 = torch.randn(100, 3)*0.2 +(torch.rand(1, 3).expand(100, 3) -0.5) * 3
#     x3 = torch.randn(100, 3)*0.2 +(torch.rand(1, 3).expand(100, 3) -0.5) * 3
#     X = torch.cat((x1,x2,x3), 0)
#     X_b = torch.cat((x1.unsqueeze(0),x2.unsqueeze(0),x3.unsqueeze(0)), 0)
#     xn  = X.norm(2,-1)

#     X[xn>1] /= ((xn[xn>1]).unsqueeze(-1).expand((xn[xn>1]).shape[0], 3) +1e-3)
#     X_b = torch.cat((X[0:100].unsqueeze(0),X[100:200].unsqueeze(0),X[200:].unsqueeze(0)), 0)
#     km = PoincareKMeans(3, min_cluster_size=20)
#     mu = km.fit(X)


#     ax.scatter(X[:100,0].numpy(), X[:100,1].numpy(), X[:100,2].numpy())
#     ax.scatter(X[100:200,0].numpy(), X[100:200,1].numpy(), X[100:200,2].numpy())
#     ax.scatter(X[200:,0].numpy(), X[200:,1].numpy(), X[200:,2].numpy())
#     ax.scatter(mu[:,0].numpy(),mu[:,1].numpy(),mu[:,2].numpy(),label="Poincare barycenter",
#                marker="s", c="red", s=100.)
#     ax.scatter(X_b.mean(1)[:,0], X_b.mean(1)[:,1],X_b.mean(1)[:,2],label="Euclidean barycenter",
#                marker="s", c="green", s=100.)
#     ax.legend()
#     plt.show()
#     print(km.predict(X))

# if __name__ == "__main__":
#     test()
