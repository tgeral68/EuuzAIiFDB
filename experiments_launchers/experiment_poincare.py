import argparse
import tqdm
import os 

import torch
from function_tools import pytorch_categorical
from torch.utils.data import DataLoader

import random 
import numpy as np

from os.path import join

from embedding_tools.poincare_embeddings_graph_multi import RiemannianEmbedding as PEmbed

from data_tools import corpora_tools, corpora, data_tools, logger
from evaluation_tools import evaluation, callback_tools

from optim_tools import optimizer
from kmeans_tools import kmeans_hyperbolic as kmh

parser = argparse.ArgumentParser(description='Start an experiment')


parser.add_argument('--lr', dest="lr", type=float, default=1,
                    help="learning rate for embedding")
parser.add_argument('--alpha', dest="alpha", type=float, default=1e-2,
                    help="alpha for embedding")
parser.add_argument('--beta', dest="beta", type=float, default=1,
                    help="beta for embedding")
parser.add_argument('--n-centroid', dest="n_centroid", type=int, default=2,
                    help="number of centroids for KMeans algorithm")
parser.add_argument('--dataset', dest="dataset", type=str, default="karate",
                    help="dataset to use for the experiments")
parser.add_argument('--walk-lenght', dest="walk_lenght", type=int, default=20,
                    help="size of random walk")
parser.add_argument('--cuda', dest="cuda", action="store_true", default=False,
                    help="using GPU for operation")
parser.add_argument('--epoch', dest="epoch", type=int, default=100,
                    help="number of iteration")
parser.add_argument('--id', dest="id", type=str, default="0",
                    help="identifier of the experiment")
parser.add_argument('--precompute-rw', dest='precompute_rw', type=int, default=1,
                    help="number of random path to precompute (for faster embedding learning) if negative \
                        the random walks is computed on flight")
parser.add_argument('--context-size', dest="context_size", type=int, default=5,
                    help="size of the context used on the random walk")
parser.add_argument("--negative-sampling", dest="negative_sampling", type=int, default=10,
                    help="number of negative samples for loss O2")
parser.add_argument("--embedding-optimizer", dest="embedding_optimizer", type=str, default="exphsgd", 
                    help="the type of optimizer used for learning poincaré embedding")
parser.add_argument("--size", dest="size", type=int, default=2,
                    help="dimenssion of the ball")
parser.add_argument("--batch-size", dest="batch_size", type=int, default=512,
                    help="Batch number of elements")
parser.add_argument("--seed", dest="seed", type=int, default=42,
                    help="the seed used for sampling random numbers in the experiment")  
parser.add_argument('--force-rw', dest="force_rw", action="store_false", default=True,
                    help="if set will automatically compute a new random walk for the experiment")           
args = parser.parse_args()


# set the seed for random sampling
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(a=args.seed)

dataset_dict = { "karate": corpora.load_karate,
            "football": corpora.load_football,
            "dblp": corpora.load_dblp,
            "books": corpora.load_books,
            "polblog": corpora.load_polblogs,
            "adjnoun": corpora.load_adjnoun,
            "wikipedia": corpora.load_wikipedia
          }

optimizer_dict = {"addhsgd": optimizer.PoincareBallSGDAdd,
                    "exphsgd": optimizer.PoincareBallSGDExp,
                    "hsgd": optimizer.PoincareBallSGD}


print("The following options are use for the current experiment ", args)
# check if dataset exists

if(args.dataset not in dataset_dict):
    print("Dataset " + args.dataset + " does not exist, please select one of the following : ")
    print(list(dataset_dict.keys()))
    quit()

if(args.embedding_optimizer not in optimizer_dict):
    print("Optimizer " + args.embedding_optimizer + " does not exist, please select one of the following : ")
    print(list(optimizer_dict.keys()))
    quit()


print("Loading Corpus ")
D, X, Y = dataset_dict[args.dataset]()
print("Creating dataset")
# index of examples dataset
dataset_index = corpora_tools.from_indexable(torch.arange(0,len(D),1).unsqueeze(-1))
print("Dataset Size -> ", len(D))



D.set_path(False)

# negative sampling distribution
frequency = D.getFrequency()**(3/4)
frequency[:,1] /= frequency[:,1].sum()
frequency = pytorch_categorical.Categorical(frequency[:,1])

# random walk dataset
d_rw = D.light_copy()

general_conf = logger.JSONLogger("data/config/general.conf", mod="fill")
general_path = general_conf["path"]
log_path = general_conf["log_path"]
rw_log = logger.JSONLogger("data/config/random_walk.conf", mod="continue")

if(args.force_rw):
    key = args.dataset+"_"+str(args.context_size)+"_"+str(args.walk_lenght)+"_"+str(args.seed) 
    if(key in rw_log):

        try:
            print('Loading random walks from files')
            d_rw = torch.load(rw_log[key]["file"])
            print('Loaded')
        except:
            os.makedirs(general_path, exist_ok=True)
            d_rw.set_walk(args.walk_lenght, 1.0)
            d_rw.set_path(True)
            d_rw = corpora.ContextCorpus(d_rw, context_size=args.context_size, precompute=args.precompute_rw)
            torch.save(d_rw, join(general_path, key+".t7"))
            rw_log[key] = {"file":join(general_path, key+".t7"), 
                            "context_size":args.context_size,
                            "walk_lenght": args.walk_lenght,
                            "precompute_rw": args.precompute_rw,
                            "dataset": args.dataset
                            }            
    else:
        os.makedirs(general_path, exist_ok=True)
        d_rw.set_walk(args.walk_lenght, 1.0)
        d_rw.set_path(True)
        d_rw = corpora.ContextCorpus(d_rw, context_size=args.context_size, precompute=args.precompute_rw)
        torch.save(d_rw, join(general_path, key+".t7"))
        rw_log[key] = {"file":join(general_path, key+".t7"), 
                       "context_size":args.context_size, 
                       "walk_lenght": args.walk_lenght,
                       "precompute_rw": args.precompute_rw,
                       "dataset": args.dataset}
else:
    d_rw.set_walk(args.walk_lenght, 1.0)
    d_rw.set_path(True)
    d_rw = corpora.ContextCorpus(d_rw, context_size=args.context_size, precompute=args.precompute_rw)   

# neigbhor dataset
d_v = D.light_copy()
d_v.set_walk(1, 1.0)

print("Merging dataset")
embedding_dataset = corpora_tools.zip_datasets(dataset_index,
                                                corpora_tools.select_from_index(d_v, element_index=0),
                                                d_rw
                                                )
training_dataloader = DataLoader(embedding_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True,
                            num_workers=4,
                            collate_fn=data_tools.PadCollate(dim=0),
                            drop_last=False
                    )

# create folder and files for logs
os.makedirs(join(log_path,args.id+"/"), exist_ok=True)
logger_object = logger.JSONLogger(join(log_path,args.id+"/log.json"))
logger_object.append(vars(args))

def log_callback_conductance(embeddings, adjancy_matrix, n_centroid):
    kmeans = kmh.PoincareKMeans(n_centroid)
    kmeans.fit(embeddings)
    i =  kmeans.predict(embeddings)
    r = torch.arange(0, i.size(0), device=i.device)
    prediction = torch.zeros(embeddings.size(0), n_centroid)
    prediction[r,i] = 1
    return {"conductance":evaluation.mean_conductance(prediction, adjancy_matrix)}

alpha, beta = args.alpha, args.beta
embedding_alg = PEmbed(len(embedding_dataset), size=args.size, lr=args.lr, cuda=args.cuda, negative_distribution=frequency,
                        optimizer_method=optimizer_dict[args.embedding_optimizer])

cf =callback_tools.generic_callback({"embeddings": embedding_alg.get_PoincareEmbeddings}, 
                                {"adjancy_matrix":X, "n_centroid": args.n_centroid},
                                log_callback_conductance)
embedding_alg.fit(training_dataloader, alpha=alpha, beta=beta, max_iter=args.epoch,
                  negative_sampling=args.negative_sampling, log_callback=cf, logger=logger_object)

embeds = embedding_alg.get_PoincareEmbeddings().cpu()




# evaluate performances and append it into the log file
total_accuracy, stdmx, stdmean, all_std = evaluation.unsupervised_poincare_eval(embeds, D.Y, args.n_centroid, verbose=False)
print("\nPerformances  kmeans-> " ,
    total_accuracy
)
logger_object.append({"accuracy_kmeans": total_accuracy})

# saving embeddings
torch.save(embeds, join(log_path, args.id+"/embeddings.t7"))
