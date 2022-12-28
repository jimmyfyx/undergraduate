import torch
import hw4_utils

def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [N, 2].
        init_c: initial centroids, shape [2, 2]. Each row is a centroid.
    
    Return:
        c: shape [2, 2]. Each row is a centroid.
    """

    if X is None:
        X, init_c = hw4_utils.load_data()
    
    k = 2               # number of cluster
    n = X.shape[0]      # number of data points
    
    prev_centroid = init_c.clone()  # record the previous centroids
    for num_itr in range(n_iters):
        cluster_0 = torch.tensor([[0, 0]])
        cluster_1 = torch.tensor([[0, 0]])
        for i in range(n):
            # for every sample x assign x to the closet cluster
            closet_cluster = -1
            min_dist = float('inf')
            for j in range(k):
                dist = torch.sqrt((X[i][0] - init_c[j][0])**2 + (X[i][1] - init_c[j][1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closet_cluster = j
            if closet_cluster == 0:
                cluster_0 = torch.cat((cluster_0, torch.tensor([[X[i][0], X[i][1]]])), 0)
            else:
                cluster_1 = torch.cat((cluster_1, torch.tensor([[X[i][0], X[i][1]]])), 0)

        # visualize
        hw4_utils.vis_cluster(init_c, cluster_0[1:, :], cluster_1[1:, :])

        # calculate new centroids
        init_c[0] = torch.sum(cluster_0, 0) / (cluster_0.shape[0] - 1)
        init_c[1] = torch.sum(cluster_1, 0) / (cluster_1.shape[0] - 1)
        
        # decide whether to break
        break_flag = True
        for i in range(k):
            if prev_centroid[i][0] != init_c[i][0] or prev_centroid[i][1] != init_c[i][1]:
                break_flag = False
        
        if break_flag == True:
            print('Steps to converge: {num}'.format(num=num_itr))
            print(init_c)
            cost = 0
            for i in range(cluster_0.shape[0]):
                cost += torch.sqrt((cluster_0[i][0] - init_c[0][0])**2 + (cluster_0[i][1] - init_c[0][1])**2)
            for i in range(cluster_1.shape[0]):
                cost += torch.sqrt((cluster_1[i][0] - init_c[1][0])**2 + (cluster_1[i][1] - init_c[1][1])**2)
            print(cost)
            break

        prev_centroid = init_c.clone()
    
    return init_c


k_means()
    
    
