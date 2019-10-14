# Learning embeddings
echo ' Learning embeddings '
python3.7  experiments_launchers/experiment_poincare.py --id 'karate_example'

# Evaluate embeddings unsupervised way using 20 
# kmeans (criterion to select experiment are given in the paper)
# for getting best evaluation you can use the notebook
echo ' Start unsupervised evaluation'
python3.7  experiments_launchers/evaluation_unsupervised_poincare.py --n 20 --id 'karate_example'
echo ' Start supervised evaluation'
python3.7  experiments_launchers/evaluation_supervised_poincare.py --id 'karate_example'

# scatter the embeddings 
echo ' Plot the embeddings and centroids '
python3.7  experiments_launchers/visualisation_poincare_kmeans.py --id 'karate_example'

# Learning embeddings using euclidean space
echo ' Learning embeddings using euclidean manifold'
python3.7  experiments_launchers/experiment_euclidean.py --id 'karate_example_euclidean'

# Evaluate embeddings unsupervised way using 20 
# kmeans (criterion to select experiment are given in the paper)
# for getting best evaluation you can use the notebook
echo ' Start unsupervised evaluation on euclidean manifold'
python3.7  experiments_launchers/evaluation_unsupervised_euclidean.py --n 20 --id 'karate_example_euclidean'
