import os

import matplotlib.pyplot as plt
import numpy as np

# plot embeddings into disc
def plot_embeddings(z, colors=None, centroids=None, save_folder=".", file_name="default.png",
                    marker='.', s=2000.):
    fig = plt.figure(" Embeddings ", figsize=(20, 20))
    fig.patch.set_visible(False)

    # draw circle
    theta = np.linspace(0, 2*np.pi, 100)

    r = np.sqrt(1.0)

    x1 = r*np.cos(theta)
    x2 = r*np.sin(theta)
    plt.plot(x1, x2)

    # plotting embeddings
    for q in range(len(z)):
        c_color = [colors[q]] if(colors is not None) else "red"
        plt.scatter(z[q][0].item(), z[q][1].item(), c=c_color, marker=marker, s=s)    
    if(centroids is not None):
        plt.scatter(centroids[:, 0], centroids[:,1],marker='D', s=800.)      
    filepath = os.path.join(save_folder, file_name)
    plt.axis('off')
    plt.savefig(filepath, format="png")
    plt.close()
