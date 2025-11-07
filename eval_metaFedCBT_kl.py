import numpy as np

from config import *
from helper import *
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from meta_model import MetaDGN
import networkx as nx
def pagerank(adj_matrix, d=0.85, tol=1e-6, max_iter=100):
    graph = nx.from_numpy_array(adj_matrix,create_using=nx.DiGraph)
    # G = nx.DiGraph()
    # G.add_weighted_edges_from(grap  h.edges(data=True))
    # pr= nx.pagerank(graph)
    pr=nx.effective_size(graph,weight='weight')
    #pr=nx.clustering(graph,weight='weight')
    # # adj_matrix: adjacency matrix of the web graph
    # # d: damping factor (default: 0.85)
    # # tol: tolerance for convergence (default: 1e-6)
    # # max_iter: maximum number of iterations (default: 100)
    #
    # n = adj_matrix.shape[0]
    # # Initialize PageRank values
    # pr = np.ones(n) / n
    # # Compute out-degree of each node
    # out_degree = np.sum(adj_matrix, axis=1)
    # # Make sure there are no nodes with out-degree 0
    # out_degree[out_degree == 0] = 1
    # # Normalize adjacency matrix by out-degree
    # adj_matrix = adj_matrix / out_degree[:, np.newaxis]
    #
    # for i in range(max_iter):
    #     # Compute new PageRank values
    #     new_pr = (1 - d) / n + d * np.sum(adj_matrix * pr, axis=0)
    #     # Check convergence
    #     # if np.max(np.abs(new_pr - pr)) < tol:
    #     #     break
    #     pr = new_pr

    pr_list=np.array(list(pr.values()))
    prm= pr_list / pr_list.sum(axis=0)
    return prm
Setup_name='federation-scaffold'
## federation-fednyn for FedNyn
#federation-fednova for FedNova
#federation-scaffold for scaffold
#federation-moon-local for MOOn
#federation-prox-local for FedProx
result = []
for i in range(n_folds):
    test_result = []
    print("Fold ",i)
    print("---------")
    test_data = np.load('{}fold{}/fold{}_test_data.npy'.format(Path_input, i, i))
    for data in test_data:
        for view in range(N_views):
            data_view = data[:, :, view]
            cbt_probs = np.array(pagerank(data_view))
            test_result.append(cbt_probs)
    ground_truth=np.mean(test_result, axis=0)
    print(ground_truth)

    for k in range(number_of_samples):
        result_list = []
        for j in range(random_num):
            cbt_train = np.load('{}fold{}_{}_cli_{}_{}_cbt.npy'.format('output/' + Dataset_name + '/' + Setup_name + '/', i,j, i, k, Setup_name))
            cbt_probs = np.array(pagerank(cbt_train))
            temp = np.mean(np.abs(cbt_probs - ground_truth))
            result_list.append(temp)
            result.append(temp)
            #print(temp)
        #print(result_list)
        print(np.mean(result_list,axis=0))


        # tagert_dis=np.array([[pagerank(data[:,:,i]) for i in range(N_views)] for data in test_data])
        # kl_loss = np.array([[abs((target_probs1 * np.log2(target_probs1 / cbt_probs)).sum()) for target_probs1 in target] for target in tagert_dis])
        # print(kl_loss.mean())
        # target= test_data.mean(axis=(0,3))
        # target_probs=pagerank(target)
        # kl_loss=abs((target_probs * np.log2(target_probs/ cbt_probs)).sum())
        # print(kl_loss)
        # kl_list = []
        # for data in test_data:
        #     for view in range(N_views):
        #         data_view = data[:, :, view]
        #         cbt_probs = np.array(pagerank(data_view))
        #         result_list.append(cbt_probs)
    print(np.mean(result_list, axis=0))
print(np.mean(result, axis=0))
        #     target = data.mean(axis=0)
        #     target_probs = pagerank(target)
        #     kl_loss = abs((target_probs * np.log2(target_probs / cbt_probs)).sum())
        #     kl_list.append(kl_loss)
        # # kl_loss=abs((target_probs * np.log2(target_probs/ cbt_probs)).sum())
        # kl_list = np.array(kl_list)
        # print(kl_list.mean(), kl_list.sum())
#
#         cbt=np.zeros(cbt_train.shape)
#         target=np.zeros(cbt_train.shape)
#         cbt[np.where(cbt_train>0)]=1
#
#         graph = nx.from_numpy_array(cbt_train)
#         n=nx.get_edge_attributes(graph,'weight')
#
#         cluster_cbt=nx.clustering(graph,weight=graph.edges(data=True))
#         ens = nx.effective_size(graph,weight=graph.edges(data=True))
#
#         # topological measurement

#         cbt_diss=np.array([[data[:,:,j] for j in range(N_views)]for data in test_data]).mean(axis=(0,1))
# #         target[np.where(cbt_diss > 0)] = 1
#         cbt_diss=pagerank(target)
#         # #cbt_dist = cbt_dis.sum(axis=0)
#         cbt_probs = cbt_dis / cbt_dis.sum(axis=0)
#         # kl_loss=[]
#         target_probs1 = cbt_diss / cbt_diss.sum(axis=0)
#         kl_loss=abs((target_probs1 * np.log2(target_probs1 / cbt_probs)).sum())
#         print(kl_loss)
#         # for target1 in cbt_diss:
#         #     target_probs1 = target1/ target1.sum(axis=0)
#         #     kl_loss.append(abs((target_probs1 * np.log2(target_probs1/cbt_probs)).sum()))
#         # print(np.mean(kl_loss))
# # Example graph (5 nodes)
# graph = nx.Graph()
# graph.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
#
# # Calculate effective node size
# ens = effective_node_size(graph)
#
# # Print the result
# print(ens)   # [0.283, 0.747, 0.517, 0.408, 0.225]

