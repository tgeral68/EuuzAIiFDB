import torch
from function_tools import poincare_function as pf
from function_tools import poincare_alg as pa
from function_tools import euclidean_function as ef
import tqdm
from kmeans_tools.kmeans_hyperbolic import PoincareKMeans
from collections import Counter
import numpy as np
import math
import itertools


############################ NEW #####################
# SCORING FUNCTION
def _precision_at(prediction, ground_truth, at=5):
    prediction_value, prediction_index = (-prediction).sort(-1)
    trange = torch.arange(len(prediction)).unsqueeze(-1).expand(len(prediction), at).flatten()
    indexes = prediction_index[:,:at].flatten()
    score = ((ground_truth[trange, indexes]).float().view(len(prediction), at)).sum(-1)/at
    return score.mean().item()

def nmi(prediction, ground_truth):
    N = prediction.size(0)
    # the number of clusters
    K = prediction.size(-1)

    I = {i for i in range(len(prediction))}

    PN = []
    GN = []
    den = 0
    for i in range(K):
        PN.append(set(prediction[:, i].nonzero().flatten().tolist()))
        GN.append(set(ground_truth[:, i].nonzero().flatten().tolist()))
        if(len(PN[-1]) != 0):
            den += len(PN[-1]) * math.log(len(PN[-1])/N)
        if(len(GN[-1]) != 0):
            den += len(GN[-1]) * math.log(len(GN[-1])/N)
    num = 0
    for a in PN:
        for b in GN:
            N_ij = len(a.intersection(b))
            if(N_ij != 0):
                num += N_ij * math.log((N_ij * N)/(len(a) *len(b) ))
    
    return -2 * (num/den)

def mean_conductance(prediction, adjency_matrix):
    N = prediction.size(0)
    # the number of clusters
    K = prediction.size(-1)
    # print(K)

    I = {i for i in range(len(prediction))}

    score = 0
    for c in range(K):
        # print(prediction[:, c].nonzero().flatten())
        c_nodes = set(prediction[:, c].nonzero().flatten().tolist())
        nc_nodes = I - c_nodes
        cut_score_a = 0
        for i in c_nodes:
            cut_score_a += len(set(adjency_matrix[i]) - c_nodes)
            # for j in nc_nodes:
            #     if(j in adjency_matrix[i]):
            #         cut_score_a += 1
        cut_score_b = 0
        for i in c_nodes:
            cut_score_b += len(adjency_matrix[i])

        cut_score_c = 0
        for i in nc_nodes:
            cut_score_c += len(adjency_matrix[i])
        if(cut_score_b==0 or cut_score_c ==0):
            score += 0 
        else:
            score += cut_score_a/(min(cut_score_b, cut_score_c))
    
    return score/K

def topk_conductance(prediction, adjency_matrix, k=1):
    N = prediction.size(0)
    # the number of clusters
    K = prediction.size(-1)
    # print(K)

    I = {i for i in range(len(prediction))}

    score = []
    for c in range(K):
        # print(prediction[:, c].nonzero().flatten())
        c_nodes = set(prediction[:, c].nonzero().flatten().tolist())
        nc_nodes = I - c_nodes
        cut_score_a = 0
        for i in c_nodes:
            cut_score_a += len(set(adjency_matrix[i]) - c_nodes)
            # for j in nc_nodes:
            #     if(j in adjency_matrix[i]):
            #         cut_score_a += 1
        cut_score_b = 0
        for i in c_nodes:
            cut_score_b += len(adjency_matrix[i])

        cut_score_c = 0
        for i in nc_nodes:
            cut_score_c += len(adjency_matrix[i])
        if(cut_score_b==0 or cut_score_c ==0):
            pass
        else:
            score.append(cut_score_a/(min(cut_score_b, cut_score_c)))
    
    return score/K
class PrecisionScore(object):
    def __init__(self, at=5):
        self.at = at

    def __call__(self, x, y):
        return _precision_at(x, y, at=self.at)


# META EVAL
class CrossValEvaluation(object):
    def __init__(self, embeddings, ground_truth, nb_set=5, algs_object=None):
        self.algs_object = algs_object
        self.z = embeddings
        self.gt = ground_truth
        self.nb_set = nb_set
        # split set
        subset_index = torch.randperm(len(self.z))
        nb_value = len(self.z)//nb_set
        self.subset_indexer = [subset_index[nb_value *i:min(nb_value * (i+1), len(self.z))] for i in range(nb_set)]
        self.all_algs = []
        pb = tqdm.trange(len(self.subset_indexer))

        for i, test_index in zip(pb, self.subset_indexer):
            # create train dataset being concatenation of not current test set
            train_index = torch.cat([ subset for ci, subset in enumerate(self.subset_indexer) if(i!=ci)], 0)
            
            # get embeddings sets
            train_embeddings = self.z[train_index]
            test_embeddings  = self.z[test_index]

            # get ground truth sets
            train_labels = self.gt[train_index]
            test_labels  =  self.gt[test_index]
            algs = self.algs_object(self.gt.size(-1))
            algs.fit(train_embeddings, Y=train_labels)
            self.all_algs.append(algs)
        
    def get_score(self, scoring_function):
        scores = []
        pb = tqdm.trange(len(self.subset_indexer))
        for i, test_index in zip(pb, self.subset_indexer):
            # create train dataset being concatenation of not current test set
            train_index = torch.cat([ subset for ci, subset in enumerate(self.subset_indexer) if(i!=ci)], 0)
            
            # get embeddings sets
            train_embeddings = self.z[train_index]
            test_embeddings  = self.z[test_index]

            # get ground truth sets
            train_labels = self.gt[train_index]
            test_labels  =  self.gt[test_index]

            # must give the matrix of scores
            prediction = self.all_algs[i].probs(test_embeddings)

            set_score = scoring_function(prediction, test_labels)
            scores.append(set_score)
        return scores


# in the following function we perform prediction using disc product
# Z, Y, pi, mu, sigma are list of tensor with the size number of disc

def supervised_poincare_eval(z, y, n_centroid, nb_set=5, verbose=True):
    n_example = len(z)
    subset_index = torch.randperm(n_example)
    nb_value = n_example//nb_set
    I_CV = [subset_index[nb_value *i:min(nb_value * (i+1), n_example)] for i in range(nb_set)]
    # print(I_CV)
    acc_total = 0.
    for i, test_index in enumerate(I_CV):
        # create train dataset
        train_index = torch.cat([ subset for ci, subset in enumerate(I_CV) if(i!=ci)],0)
        Z_train = z[train_index]
        Y_train = torch.LongTensor([y[ic.item()] for ic in train_index])

        #create test datase
        Z_test = z[test_index]
        Y_test = torch.LongTensor([y[ic.item()] for ic in test_index])      
        
        if(verbose):
            print("Set "+str(i)+" :")
            print("\t train size -> "+str(len(Z_train)))
            print("\t test size -> "+str(len(Z_test)))
            print("Obtaining centroids for each classes")
        
        min_label = Y_train.min().item()
        max_label = Y_train.max().item()


        centroids = []
        for i in range(n_centroid):
            # print((Z_train[Y_train[:,0]== (min_label + i)]).size())
            centroids.append(pa.barycenter(Z_train[Y_train[:,0]== (min_label + i)], normed=False).tolist())
        
        centroids = torch.Tensor(centroids).squeeze()
        # predicting 
        Z_test_reshape = Z_test.unsqueeze(1).expand(Z_test.size(0), n_centroid, Z_test.size(-1))
        centroid_reshape = centroids.unsqueeze(0).expand_as(Z_test_reshape)

        d2 = pf.distance(Z_test_reshape, centroid_reshape)**2

        predicted_labels = d2.min(-1)[1] + min_label

        acc = (predicted_labels == Y_test.squeeze()).float().mean()
        acc_total += acc.item()
    return acc_total/(len(I_CV))


def supervised_euclidean_eval(z, y, mu, nb_set=5, verbose=True):
    n_example = len(z)
    n_distrib = len(mu)
    subset_index = torch.randperm(n_example)
    nb_value = n_example//nb_set
    I_CV = [subset_index[nb_value *i:min(nb_value * (i+1), n_example)] for i in range(nb_set)]
    # print(I_CV)
    acc_total = 0.
    for i, test_index in enumerate(I_CV):
        # create train dataset
        train_index = torch.cat([ subset for ci, subset in enumerate(I_CV) if(i!=ci)],0)
        Z_train = z[train_index]
        Y_train = torch.LongTensor([y[ic.item()] for ic in train_index])

        #create test datase
        Z_test = z[test_index]
        Y_test = torch.LongTensor([y[ic.item()] for ic in test_index])      
        
        if(verbose):
            print("Set "+str(i)+" :")
            print("\t train size -> "+str(len(Z_train)))
            print("\t test size -> "+str(len(Z_test)))
            print("Obtaining centroids for each classes")
        
        from function_tools import poincare_alg as pa
        min_label = Y_train.min().item()
        max_label = Y_train.max().item()


        centroids = []
        for i in range(n_distrib):
            print(Z_train[Y_train[:,0]== (min_label + i)].size())
            centroids.append((Z_train[Y_train[:,0]== (min_label + i)].mean(0)).tolist())
        
        centroids = torch.Tensor(centroids).squeeze()
        # predicting 
        Z_test_reshape = Z_test.unsqueeze(1).expand(Z_test.size(0), n_distrib, Z_test.size(-1))
        centroid_reshape = centroids.unsqueeze(0).expand_as(Z_test_reshape)

        d2 = ef.distance(Z_test_reshape, centroid_reshape)**2

        predicted_labels = d2.min(-1)[1] + min_label

        acc = (predicted_labels == Y_test.squeeze()).float().mean()
        acc_total += acc.item()
        print(acc)
    return acc_total/(len(I_CV))

def unsupervised_poincare_eval(z, y, n_centroid, verbose=False):
    n_example = len(z)
    y = torch.LongTensor([y[i][0]-1 for i in range(len(y))])

    # first getting the pdf for each disc distribution
    kmeans = PoincareKMeans(n_centroid)
    kmeans.fit(z)
    associated_distrib =  kmeans.predict(z)

    label = associated_distrib.numpy()
    label_source = y.numpy()

    std =   kmeans.getStd(z)
    if(n_centroid <= 6):
        return accuracy_small_disc_product(label, label_source, n_centroid), std.max(), std.mean(), std
    else:
        return accuracy_huge_disc_product(label, label_source, n_centroid),std.max(), std.mean(), std




def unsupervised_euclidean_eval(z, y, n_centroid, verbose=False):
    n_example = len(z)

    y = torch.LongTensor([y[i][0]-1 for i in range(len(y))])
    from sklearn.cluster import KMeans

    # kmean euclidean
    kmeans = KMeans(n_centroid, n_init=1)
    kmeans.fit(z.numpy())
    associated_distrib =  kmeans.predict(z.numpy())
    centroids = torch.Tensor(kmeans.cluster_centers_)

    
    N, K, D = z.shape[0], centroids.shape[0], z.shape[1]
    centroids = centroids.unsqueeze(0).expand(N, K, D)
    x = z.unsqueeze(1).expand(N, K, D)
    dst =(centroids-x).norm(2,-1)**2
    value, indexes = dst.min(-1)
    stds = []
    for i in range(n_centroid):
        stds.append(value[indexes==i].sum())
    std  = torch.Tensor(stds)
    label = associated_distrib
    label_source = y.numpy()

    if(n_centroid <= 6):
        return accuracy_small_disc_product(label, label_source, n_centroid), std.tolist()
    else:
        return accuracy_huge_disc_product(label, label_source, sources_number), std.tolist()
def accuracy_small_disc_product(label, label_source, sources_number):
    combinations = []
    zero_fill_comb = np.zeros(len(label))

    Numbers =  np.arange(0, sources_number)
    numerotations = list(itertools.permutations(Numbers))

    # print("zeroçfcom", len(label))

    for i in range(0,math.factorial(sources_number)):
        combinations.append(zero_fill_comb)


    combinations = np.array(combinations)
    numerotations = np.array(numerotations)


    for i in range(0,len(combinations)):
         combinations[i] = label_source.copy()



    # Calcul des tableaux permutés
    for i in range (0,len(numerotations)):

        # print('i',i)
        # print('numerotation\n', numerotations[i])
        for j in range(0,len(combinations[i])):

            for q in range(0,len(Numbers)):
                if(combinations[i][j]== Numbers[q]):
                    combinations[i][j] = numerotations[i][q]
                    break



    # print('Combinations after permutations\n',combinations)

    result = np.zeros(len(combinations[0]))

    # print('Len result',len(combinations[:,0]))


    result_percentage = []

    for u in range(0,len(combinations[:,0])):

        result_combination = (combinations[u]-label)

        # print('result combination', result_combination)

        np.append(result, result_combination)



        result_int = (sum(1 for i in result_combination if i == 0) / len(label_source)) * 100

        # print('sum(1 for i in result_combination if i == 0)',sum(1 for i in result_combination if i == 0))

        result_percentage.append(result_int)



    # print('result',result_percentage)
    return max(result_percentage)

def accuracy_huge_disc_product(label, label_source, sources_number):

    numerotation_initial = np.zeros(sources_number, dtype=int)

    numerotation_initial = numerotation_initial - 1

    # print('Numerotation initial\n', numerotation_initial)

    number_data_per_cluster = np.zeros(sources_number, dtype=int)

    priority_clusters = np.zeros(sources_number, dtype=int)

    for j in range(0, len(priority_clusters)):
        priority_clusters[j] = j

    # print('Priority Cluster\n', priority_clusters)

    # Pour chaque cluster calculé
    for i in range(0, sources_number):
        for j in range(0, len(label)):
            if (label[j] == i):
                # On calcul le nombre de données par Cluster
                number_data_per_cluster[i] = number_data_per_cluster[i] + 1

                # Pour chaque donnée qui appartient à ce Cluster
                # On va voir le cluster de la verite de terrain et compter

    # print('Number Data per cluster\n',number_data_per_cluster)

    # On va classer les clusters selon le nombre de donnees qu'ils contiennent
    # Par ordre decroissant

    for q in range(0, len(priority_clusters)):
        for u in range(q + 1, len(priority_clusters)):
            if (number_data_per_cluster[priority_clusters[q]] < number_data_per_cluster[priority_clusters[u]]):
                temp = priority_clusters[q].copy()
                priority_clusters[q] = priority_clusters[u].copy()
                priority_clusters[u] = temp.copy()

    # print('Priority Clusters after\n',priority_clusters)

    # On commence par le cluster le plus prioritaire A (plus de donnnes)

    taken_or_not = []
    for i in range(0, sources_number):
        taken_or_not.append(False)

    for i in range(0, len(priority_clusters)):

        # On cherche le noeud de la verite de terrain qui apparait le plus de fois dans A
        count = np.zeros(sources_number, dtype=int)

        for j in range(0, len(label)):

            # Pour chaque donnée qui appartient à A
            if (label[j] == priority_clusters[i]):
                count[label_source[j]] = count[label_source[j]] + 1

        # print('Count for cluster',priority_clusters[i],'is\n',count)

        max_count = 0
        for q in range(0, len(count)):
            if (count[q] >= max_count and taken_or_not[q] == False):
                max_count = count[q]
                numerotation_initial[priority_clusters[i]] = q

        taken_or_not[numerotation_initial[priority_clusters[i]]] = True

    # print('Large K Guess\n', numerotation_initial)

    # Maintenant faut remplacer les numéros de label par ceux du numerotation_initial

    new_label = label.copy()

    for j in range(0, len(new_label)):
        for q in range(0, len(numerotation_initial)):
            if (new_label[j] == q):
                new_label[j] = numerotation_initial[q]
                break

    # print('New label\n')
    # for i in new_label:
    #    print(i)

    # print('Source label\n')
    # for i in label_source:
    #   print(i)

    result_combination = (new_label - label_source)

    max_result = (sum(1 for i in result_combination if i == 0) / len(label_source)) * 100

    # print('Max result large K',max_result)

    # Apres il faut faire les permutations sur les classes de tout ceux qui sont pas taken

    # untaken = []
    #
    # print('taken or not\n',taken_or_not)
    #
    # for i in range(0,len(taken_or_not)):
    #     if taken_or_not[i] == False:
    #         print(i)
    #         untaken.append(i)
    #
    # print('untaken\n',untaken)
    #
    # numerotations = list(itertools.permutations(untaken))
    #
    # print('Numerotations', numerotations)

    return max_result
