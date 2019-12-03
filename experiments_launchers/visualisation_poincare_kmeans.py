import argparse
import os
from os.path import join

import torch



import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import numpy as np


from data_tools import corpora_tools, corpora, data_tools, logger

from visualisation_tools import poincare_plot

from kmeans_tools import kmeans_hyperbolic as kmh



parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')

parser.add_argument('--id', dest="id", type=str, default="0",
                    help="embeddings location id")
parser.add_argument('--n', dest="n", type=int, default=1,
                    help="number of times to perform kmeans")              
args = parser.parse_args()


dataset_dict = { "karate": corpora.load_karate,
            "football": corpora.load_football,
            "dblp": corpora.load_dblp,
            "books": corpora.load_books,
            "polblog": corpora.load_polblogs,
            "adjnoun": corpora.load_adjnoun
          }

# loading the configuration file
general_conf = logger.JSONLogger("data/config/general.conf", mod="continue")
# get the folder of the experiment
folder_xp = join(general_conf["log_path"], args.id)
# load the log file of the experiment
log_xp = logger.JSONLogger(join(folder_xp,"log.json"), mod="continue")

# reading configuration of the xp
dataset_name = log_xp["dataset"]
n_centroid = log_xp["n_centroid"]

if(dataset_name not in dataset_dict):
    print("Dataset " + dataset_name + " does not exist, please select one of the following : ")
    print(list(dataset_dict.keys()))
    quit()


print("Loading dataset : ", dataset_name)
D, X, Y = dataset_dict[dataset_name]()

results = []
std_kmeans = []
representations = torch.load(os.path.join(folder_xp,"embeddings.t7"))

kmeans = kmh.PoincareKMeans(n_centroid)
kmeans.fit(representations)
gt_colors = []
pr_colors = []

unique_label = np.unique(sum([ y for k, y in D.Y.items()],[]))

prediction = kmeans.predict(representations)

for i in range(len(D.Y)):
    gt_colors.append(plt_colors.hsv_to_rgb([D.Y[i][0]/(len(unique_label)),0.5,0.8]))
    pr_colors.append(plt_colors.hsv_to_rgb([prediction[i].item()/(len(unique_label)),0.5,0.8]))

# plotting unsupervised prediction
poincare_plot.plot_embeddings(representations, centroids=kmeans.centroids,
                              colors=pr_colors, save_folder=folder_xp, 
                              file_name="unsupervised_prediction_embeddings.png")

# plotting ground truth
poincare_plot.plot_embeddings(representations, colors=gt_colors, save_folder=folder_xp,
                              file_name="ground_truth_embeddings.png")

ground_truth = torch.LongTensor([[ 1 if(y+1 in Y[i]) else 0 for y in range(n_centroid)] for i in range(len(X))])
kmeans = kmh.PoincareKMeans(n_centroid)
kmeans.fit(representations, ground_truth)
prediction = kmeans.predict(representations)
pr_colors = []
for i in range(len(D.Y)):
    pr_colors.append(plt_colors.hsv_to_rgb([prediction[i].item()/(len(unique_label)),0.5,0.8]))

# plotting supervised prediction
poincare_plot.plot_embeddings(representations, centroids=kmeans.centroids,
                              colors=pr_colors, save_folder=folder_xp, 
                              file_name="supervised_prediction_embeddings.png")