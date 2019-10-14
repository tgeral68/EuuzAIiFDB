from torch.utils.data import Dataset
import torch
import tqdm
import random

########################## DATASET OBJECT ############################
class ZipDataset(Dataset):
    def __init__(self, *args):
        self.datasets = args

        # checking all dataset have the same length
        self.len_dataset = None
        for dataset in self.datasets:
            if(self.len_dataset is None):
                self.len_dataset = len(dataset)
            else:
                assert(self.len_dataset == len(dataset))
    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        return sum([dataset[index] for dataset in self.datasets] ,())

class IndexableDataset(Dataset):
    def __init__(self, L):
        self.indexable_structure = L
    
    def __len__(self):
        return len(self.indexable_structure)
    
    def __getitem__(self, index):
        return (self.indexable_structure[index],)
class SelectFromPermutation(Dataset):
    def __init__(self, dataset, permutation):
        self.dataset = dataset
        self.permutation = permutation
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        permutation_index = self.permutation[index]
        if(type(permutation_index) == tuple):
            permutation_index = permutation_index[0]
        if(type(permutation_index) != int):
            if(type(permutation_index) == torch.Tensor):
                permutation_index = int(permutation_index.item())
            elif(type(permutation_index) == float):
                permutation_index = int(permutation_index)
        res = self.dataset[permutation_index]
        if(type(res) == tuple):
            return res
        else:
            return (res,)
class SelectFromIndexDataset(Dataset):
    def __init__(self, dataset, element_index=0):
        self.indexable_structure = dataset
        self.element_index = element_index
    def __len__(self):
        return len(self.indexable_structure)
    
    def __getitem__(self, index):
        return (self.indexable_structure[index][self.element_index],)    

class KNNDataset(Dataset):
    def __init__(self, X, Y, distance, k=10, n_sample=10, alpha=None):
        # to return when call
        self.X = X
        self.n_sample = n_sample
        self.knn = KNNDataset._build( Y, distance, k, alpha)
    @staticmethod
    def _build(Y, distance, k, alpha):
        if(alpha is not None):
            Y_rebase,sl,c,tll = KNNDataset._remove_tail_labels(Y, alpha)
            print("Number of used labels for Neigborhood/Total: " +str(len(sl))+"/"+str(len(c)))
        else:
            Y_rebase = Y
        
        reverse_index = KNNDataset._reverse_index(Y_rebase)
        knn = KNNDataset._knn(Y_rebase, distance, k=k, reverse_index=reverse_index)
        return knn
    @staticmethod
    def _topk(distance, k, x, data):
        return (distance(x, data)).topk(min(k+1, len(data)))[1][1:].tolist()
    @staticmethod
    def _knn(Y, distance, k=10, reverse_index=None):
        knn = []
        print("Computing Neighbhorhood")

        for i, y in zip(tqdm.trange(len(Y)),Y):
            if(len(y) == 0):
                knn.append([i])
            else:
                # if no label has been deleted
                if(reverse_index is None):
                    knn.append(KNNDataset._topk(distance, k, y, Y))
                # selecting sub sample number of examples to compare
                else:
            
                    to_process = list(set(sum([reverse_index[l] for l in y],[])))
                    examples_to_compare = [Y[i] for i in to_process]
                    example_knn = KNNDataset._topk(distance, k, y, examples_to_compare)
                    # reindexing 
                    knn.append([ to_process[index] for index in example_knn ])
        return knn

    @staticmethod
    def _reverse_index(Y):
        print("Process inversed indexation")
        reverse_index = {}
        for index_e, labels in zip(tqdm.trange(len(Y)),Y):
            for label in labels:
                if(label not in reverse_index):
                    reverse_index[label] = set()
                reverse_index[label].add(index_e)
        return {k:list(v) for k,v in reverse_index.items()}
    @staticmethod
    def _remove_tail_labels(Y, alpha):
        max_labels = max([y[0].max().item() for y in Y])
        # we delete tail labels
        counter = torch.zeros(max_labels+1)
        print("Counting labels :")
        for tq, x in zip(tqdm.trange(len(Y)),Y):
            counter[x[0].tolist()] += 1
        v_counter, i_counter = counter.sort(0)
        print("Counting tail labels : ")
        tail_counter = 0
        tail_limit = alpha * len(Y)
        selected_labels = set()
        tail_label_limit = 0
        for tq, x in zip(tqdm.trange(len(v_counter)),v_counter):
            tail_counter += ((x * (x-1))/2).item()

            if(tail_counter >= tail_limit):
                
                print("Selecting labels from 0 to "+str(tq-1))
                tail_label_limit = tq-1
                break 
            else:
                selected_labels.add(i_counter[tq].item())
        print("Removing top labels from the data")
        Y_rebase = []
        for i in tqdm.trange(len(Y)):
            Y_rebase.append(set(Y[i][0].tolist()).intersection(selected_labels))              
        return Y_rebase, selected_labels, counter, tail_label_limit

    def __getitem__(self, index):
        rus = self.knn[index]
        if(len(rus) <= 1):
            rus = tuple([self.X[index][0] for i in range(self.n_sample)])       
        else:
            rus = tuple([self.X[rus[random.randint(1,len(rus)-1)]][0] for i in range(self.n_sample)])
        assert(len(rus)==self.n_sample)
        return rus
    
    def __len__(self):
        return len(self.X)

    @staticmethod
    def test():
        X = torch.arange(100000)
        Y = (torch.rand(100000, 5) * 500).long()
        def cosine_similarity(x, data):
            return torch.tensor([len(x.intersection(data[i]))/max(len(x) * len(data[i]),1e-8) for i in range(len(data))])
        dataset = KNNDataset(X, Y, cosine_similarity, k=10, n_sample=10, alpha=600)
        print(dataset[0])
        print(Y[0])
        print(Y[dataset.X[dataset.knn[0]][0]])
        print(Y[dataset.X[dataset.knn[0]][1]])
        print(Y[dataset.X[dataset.knn[0]][2]])
        print(Y[dataset.X[dataset.knn[0]][3]])
        print(Y[dataset.X[dataset.knn[0]][4]])
        print(Y[dataset.X[dataset.knn[0]][5]])

class SamplerDataset(Dataset):
    def __init__(self, dataset, n_sample=5, policy=None):
        self.dataset = dataset
        self.n_sample = n_sample
        self.policy = policy


    def __getitem__(self, index):
        select = self.dataset[index][0]
        if(len(select)!=0):
            if(self.policy is not None):
                probs = self.policy[select-1]/(self.policy[select-1].sum())
                distrib = torch.distributions.Categorical(probs=probs)
                rus = tuple([torch.LongTensor([select[distrib.sample()]]) for i in range(self.n_sample)])
            else:
                rus = tuple([torch.LongTensor([select[random.randint(0, len(select)-1)]]) for i in range(self.n_sample)])
        else:
            rus = tuple([torch.LongTensor([0]) for i in range(self.n_sample)])
        return rus

    def __len__(self):
        return len(self.dataset)
############################ FACTORY #############################

def zip_datasets(*args):
    return ZipDataset(*args)

def from_indexable(L):
    return IndexableDataset(L)

def select_from_index(dataset, element_index=0):
    return SelectFromIndexDataset(dataset, element_index=element_index)

def knn(X, Y, distance, k=10, n_sample=10, alpha=3000):
    return KNNDataset(X,Y, distance, k=k, n_sample=n_sample, alpha=alpha)

def sampler(dataset, n_sample=1, policy=None):
    return SamplerDataset(dataset, n_sample=n_sample, policy=policy)

def permutation_index(dataset, permutation):
    return SelectFromPermutation(dataset, permutation)