#tune code currently
import os
cuda = 0
#for yelp2018
os.system(f'python main.py --model LightGCN --loss Adaptive --augment No --init_method Normal --adaptive_method homophily --centroid_mode eigenvector --commonNeighbor_mode SC\
            --temp_tau 0.1 --alpha 0.5 --n_cluster 10 --sigma_gausse 1. --lr 0.001 --weight_decay 1e-4 --lambda1 0.1 --epsilon_GCLRec 0.1 --w_GCLRec 0.1 --k_aug 0\
            --if_visual 0 --cuda {cuda} --comment tune')
#tuning loss: change temp_tau, alpha, lr, weight_decay, lambda1, &&& adaptive coef implementation
#stop visualization currently if_visual 0