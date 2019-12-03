import argparse
import tqdm
import os

from os.path import join

import torch
import pytorch_categorical
from torch.utils.data import DataLoader

from multiprocessing import Process, Manager
from data_tools import corpora_tools, corpora, data_tools, logger
from evaluation_tools import evaluation
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

# get representations and ground truth
representations = torch.load(os.path.join(folder_xp,"embeddings.t7"))
ground_truth = torch.LongTensor([[ 1 if(y+1 in Y[i]) else 0 for y in range(n_centroid)] for i in range(len(X))])

# create the meta evaluation 
CVE = evaluation.CrossValEvaluation(representations, ground_truth, nb_set=5, algs_object=kmh.PoincareKMeans)
score = CVE.get_score(evaluation.PrecisionScore(at=1))

print("Performances : ", score)
log_xp.append({"supervised_evaluation": score})
