# running experiments if no cuda device (Nvidia GPU) please remove --cuda parameter if no GPU we recommend you to make smallest batch/learning rate and less iteration
CUDA_VISBLE_DEVICES=0 python3.7 experiments_launchers/experiment_poincare.py  --cuda  --batch-size 512 --dataset dblp  --n-centroid 5 --walk-lenght 60  --precompute-rw 6 --beta 1.  --alpha .5 --lr 25.  --negative-sampling 8  --context-size 10  --epoch 300 --embedding-optimizer exphsgd   --id "dblp-2D" --size 2 --seed 120

# create visualization unsupervised/supervised and ground truth
python3.7  experiments_launchers/visualisation_poincare_kmeans.py --id 'dblp-2D'

# evaluate in supervised setting 
python3.7  experiments_launchers/evaluation_supervised_poincare.py --id 'dblp-2D'

# evaluate in unsupervised setting 
python3.7  experiments_launchers/evaluation_unsupervised_poincare.py --n 20 --id 'dblp-2D'
