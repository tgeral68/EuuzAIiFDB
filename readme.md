# EuuzAIiFDB 
## Dependencies

- pytorch (version 1.1.0)
- pytorch_categorical
- numpy
- sklearn
- tqdm
- matplotlib

You can use conda or pip to install all the dependencies

## Start an Experiment
A complete examples is given in the script/example.sh script file. For the first experiment the script will prompt the path to save data and the log_path to store log of experiments.

## Datasets

|       Dataset         |Location                        |
|----------------|------------------------------- |
|Karate, Polblog, Adjnoun, Polbooks, Football|data folder      |
|DBLP          | Download [here](https://github.com/vwz/ComE/tree/master/data/Dblp)      |

## Learning Embeddings Parameters
|       Parameter name        | Description                      |
|----------------|------------------------------- |
|--dataset | dataset given with lowercase|
|--lr |  Learning rate (for gradient update) |
|--alpha         | O1 loss weigth   |
|--beta        | O2 loss weigth   |
|--n-centroid        | Number of centroids to perform the kmeans   |
|--walk-lenght       | Size of path used in the random walk   |
|--context-size      | Size of context (window on the path) |
|--negative-sampling | Number of negative examples to use in the loss function |
|--size      | Embeddings dimenssion size  |