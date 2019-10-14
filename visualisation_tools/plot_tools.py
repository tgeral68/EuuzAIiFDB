import math
import os

import matplotlib.pyplot as plt
import numpy as np

from function_tools import poincare_function as pf


# Plot two figures, one with the embeddings only and the ground truth community
# , the other with the associate centroids and the prediction 
def kmean_plot(z, centroids, gt_colors, pr_colors, save_folder, prefix=""):
    fig = plt.figure("Embedding-Distribution Ground truth", figsize=(20, 20))
    fig.patch.set_visible(False)
    plt.axis("off")
    theta = np.linspace(0, 2*np.pi, 100)

    r = np.sqrt(1.0)

    x1 = r*np.cos(theta)
    x2 = r*np.sin(theta)

    plt.plot(x1, x2)
    for q in range(len(z)):
        plt.scatter(z[q][0].item(), z[q][1].item(), c=[gt_colors[q]], marker='.', s=2000.)            
    filepath = os.path.join(save_folder, prefix+"ground_truth_embeddings.png")
    plt.savefig(filepath, format="png")

    fig= plt.figure("Embedding-Distribution prediction", figsize=(20, 20))
    fig.patch.set_visible(False)
    plt.axis("off")
    theta = np.linspace(0, 2*np.pi, 100)

    r = np.sqrt(1.0)

    x1 = r*np.cos(theta)
    x2 = r*np.sin(theta)

    plt.plot(x1, x2)
    for q in range(len(z)):
        plt.scatter(z[q][0].item(), z[q][1].item(), c=[pr_colors[q]], marker='.',s=2000.)
    plt.scatter(centroids[:, 0], centroids[:,1],marker='D', s=800.)            
    filepath = os.path.join(save_folder, prefix+"prediction_embeddings.png")
    plt.savefig(filepath, format="png")    
