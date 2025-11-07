import random

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from config import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance

# Create data objects for the DGN
# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
# Create data objects for the DGN
# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
# model_based fun
def cast_data(array_of_tensors, subject_type=None, flat_mask=None):
    N_ROI = array_of_tensors[0].shape[0]
    CHANNELS = array_of_tensors[0].shape[2]

    dataset = []
    for mat in array_of_tensors:  # mat.shape: (35, 35, 4)
        # Allocate numpy arrays
        edge_index = np.zeros((2, N_ROI * N_ROI))
        edge_attr = np.zeros((N_ROI * N_ROI, CHANNELS))
        x = np.zeros((N_ROI, 1))
        x = np.zeros((N_ROI, 1))
        y = np.zeros((1,))

        counter = 0
        for i in range(N_ROI):
            for j in range(N_ROI):
                edge_index[:, counter] = [i, j]
                edge_attr[counter, :] = mat[i, j]
                counter += 1

        # Fill node feature matrix (no features every node is 1)
        for i in range(N_ROI):
            x[i, 0] = 1

        # Get graph labels
        y[0] = None

        if flat_mask is not None:
            edge_index_masked = []
            edge_attr_masked = []
            for i, val in enumerate(flat_mask):
                if val == 1:
                    edge_index_masked.append(edge_index[:, i])
                    edge_attr_masked.append(edge_attr[i, :])
            edge_index = np.array(edge_index_masked).T
            edge_attr = edge_attr_masked

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        con_mat = torch.tensor(mat, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, con_mat=con_mat, y=y, label=subject_type)
        dataset.append(data)
    return dataset  # graph list

def meta_generate_cbt_median(model, train_data):
    """
        Generate optimized CBT for the training set (use post training refinement)
        Args:
            model: trained DGN model
            train_data: list of data objects
    """
    model.eval()
    cbts = []
    train_data = [d.to(device) for d in train_data]
    for data in train_data:
        cbt = model(data)
        cbts.append(np.array(cbt.cpu().detach()))
    final_cbt = torch.tensor(np.median(cbts, axis=0), dtype=torch.float32).to(device)
    return final_cbt


def mean_frobenious_distance(generated_cbt, test_data):
    """
        Calculate the mean Frobenious distance between the CBT and test subjects (all views)
        Args:
            generated_cbt: trained DGN model
            test_data: list of data objects
    """
    frobenius_all = []
    for data in test_data:
        views = data.con_mat
        for index in range(views.shape[2]):
            diff = torch.abs(views[:, :, index] - generated_cbt)
            diff = diff * diff
            sum_of_all = diff.sum()
            d = torch.sqrt(sum_of_all)
            frobenius_all.append(d)
    return sum(frobenius_all) / len(frobenius_all)


def meta_generate_subject_biased_cbts(model, train_data):
    """
        Generates all possible CBTs for a given training set.
        Args:
            model: trained DGN model
            train_data: list of data objects
    """
    model.eval()
    cbts = np.zeros((model.model_params["N_ROIs"], model.model_params["N_ROIs"], len(train_data)))
    train_data = [d.to(device) for d in train_data]
    for i, data in enumerate(train_data):
        cbt = model(data)
        cbts[:, :, i] = np.array(cbt.cpu().detach())

    return cbts

#simulate-base
def sym_noise(mean, std):
    # nosie_m=np.random.multivariate_normal(mean,cov)
    noise_m = np.random.normal(mean, std, N_Nodes * (N_Nodes - 1) // 2)
    noise_m = antiVectorize(noise_m, N_Nodes)
    return noise_m


#data processing-based
def Vectorize(data_list):
    # a list including all samples [N,N_node,N_node,N_view] transform a vectorization[N,(N_node*N_node-N_node)/2)]
    vec_subject = []
    for data in data_list:
        data = data.transpose(2, 1, 0)
        vec_subject.append(np.concatenate([data[i][np.triu_indices(data.shape[1], k=1)] for i in range(N_views)]))
    return np.array(vec_subject)


# Clears the given directory
def clear_dir(dir_name):
    for file in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file))

#plots
def plotNum(data):
    x = ["hospital"+str(i) for i in range(len(data))]  # 大写字母['A'+i for i in range(n_clusters)]
    y = [len(i) for i in data]
    sns.barplot(x=x, y=y, width=.4)
    plt.title
    plt.show()

def plotLosses(loss_table_list):
    '''
    This function plots every model's every fold's loss performance and saves them with their particular information written with their names.
    '''
    for i in range(n_folds):
        cur_loss_table = loss_table_list[i]
        if isFederated:
            fig1, ax1 = plt.subplots()
            for k in range(number_of_samples):
                loss_lst = cur_loss_table['combining_local_loss_global_data_' + str(k)]
                ax1.plot(np.arange(len(loss_lst)) + 1, loss_lst, label='Hopital{}'.format(k))
            ax1.set(xlabel='epochs', ylabel='rep loss', title='{}th Fold Combining Local Loss Global Data'.format(i))
            ax1.legend()
            ax1.grid()
            fig1.savefig('{}fold{}combining_local_loss_global_data.png'.format(Path_output, i))
            plt.show()

        fig2, ax2 = plt.subplots()
        for k in range(number_of_samples):
            loss_lst = cur_loss_table['local_loss_global_data_' + str(k)]
            ax2.plot((np.arange(len(loss_lst)) + 1), loss_lst, label='Hopital{}'.format(k))
        ax2.set(xlabel='epochs', ylabel='rep loss',
                title='{}th Fold {}th Client Local Loss Global Data {}'.format(i, k, "%.4f" % min(loss_lst)))
        ax2.grid()
        ax2.legend()
        fig2.savefig('{}fold{}_{}th_client_local_loss_global_data.png'.format(Path_output, i, k))
        plt.show()

def plotData(data):
    x = [i for i in range(len(data))]  # 大写字母['A'+i for i in range(n_clusters)]
    y = data
    sns.barplot(x=x, y=y, width=.4)
    plt.show()


def kl_matrix(kl_list):
    head = ['Distribution' + str(i) for i in range(len(kl_list))]
    fig, ax = plt.subplots(figsize=(6, 6))
    # 将元组分解为fig和ax两个变量
    im = ax.imshow(kl_list)
    # 显示图片

    ax.set_xticks(np.arange(len(head)))
    # 设置x轴刻度间隔
    ax.set_yticks(np.arange(len(head)))
    # 设置y轴刻度间隔
    ax.set_xticklabels(head)
    # 设置x轴标签'''
    ax.set_yticklabels(head)
    # 设置y轴标签'''
    for i in range(len(head)):
        for j in range(len(head)):
            text = ax.text(j, i, "{:.2f}".format(kl_list[i, j]),
                           ha="center", va="center", color="w")
    ax.set_title("KL Divergence of simulated distribution")
    # 设置题目
    fig.tight_layout()  # 自动调整子图参数,使之填充整个图像区域。
    plt.show()  # 图像展示


def random_per(data):
    for i in range(len(data)):
        per = np.random.permutation(data[i].shape[0])  # 打乱后的行号
        data[i] = data[i][per, :, :, :]  # 获取打乱后的训练数据
    return data


def show_image(img, i, k):
    img = np.repeat(np.repeat(img, 10, axis=1), 10, axis=0)
    plt.title("Fold " + str(i) + " Client " + str(k))
    plt.imshow(img)
    plt.axis('off')
    plt.colorbar()
    plt.show()
    if not os.path.exists('output/' + Dataset_name):
        os.mkdir('output/' + Dataset_name)
    if not os.path.exists('output/' + Dataset_name + '/' + Setup_name):
        os.mkdir('output/' + Dataset_name + '/' + Setup_name)
    plt.savefig('output/{}/{}/fold{}_cli_{}_{}_DGN_cbt.jpg'.format(Dataset_name, Setup_name, i, i, k, Setup_name),
                bbox_inches='tight')


# Antivectorize given vector (this gives a symmetric adjacency matrix)
def antiVectorize(vec, m):
    # Old Code
    M = np.zeros((m, m))
    M[np.triu_indices(N_Nodes, k=1)] = vec
    M = M + M.T
    M[np.diag_indices(m)] = 0
    return M


def symVectorize(M, m, vec):
    for i in range(N_views):
        M[i, :, :] = (M[i, :, :] + M[i, :, :].T) / 2
        M[i, :, :][np.diag_indices(m)] = 0
        M[i][np.round(vec, 4) == 0] = 0
    return M


def Normalization_view(view_list):
    min_view = np.min(view_list, axis=(0, 1, 2))
    max_view = np.max(view_list, axis=(0, 1, 2))
    a = (view_list - min_view) / (max_view - min_view)
    return a


def random_fun(mean, std, num):
    np.random.seed(10)
    a = np.random.normal(size=num)

    return a * std + mean


def simulate_dataset(N_Subjects, N_Nodes, N_views):
    """
        Creates random dataset
        Args:
            N_Subjects: number of subjects
            N_Nodes: number of region of interests
            N_views: number of views
        Return:
            dataset: random dataset with shape [N_Subjects, N_Nodes, N_Nodes, N_views]
    """

    features = np.triu_indices(N_Nodes)[0].shape[0]
    views = []
    for _ in range(N_views):
        view = np.random.uniform(0.1, 2, (N_Subjects, features))

        view = np.array([antiVectorize(v, N_Nodes) for v in view])
        views.append(view)
    return np.stack(views, axis=3)


def plotTSNE(X_data, Y_data, data_size, ith, isfold, isdata,plot_list=None):
    color = plt.get_cmap('PuOr', data_size)
    Y_data = np.array(Y_data)
    X_data = np.array(X_data)

    x_dr = TSNE(n_components=2, random_state=0).fit_transform(X_data)
    plt.figure()  # 创建一个画布
    for i in range(data_size):
        plt.scatter(x_dr[Y_data == i, 0], x_dr[Y_data == i, 1], label=i)  # ,c=color([i]))
    plt.legend()  # 显示图例

    if isdata:
        plt.title("TSNE Distribution {}- Hospital{}".format(Dataset_name,ith))
    elif isfold:
        plt.title("TSNE of {} - fold{}".format(Dataset_name,ith))  # 显示标题
    else:
        plt.title("TSNE of {} Cluster".format(Dataset_name))
    plt.show()


def plotPCA(X_data, Y_data, data_size, ith, isfold, isdata,plot_list=None):
    #color = plt.get_cmap('PuOr', data_size)
    Y_data = np.array(Y_data)
    X_data = np.array(X_data)

    x_dr = PCA(n_components=2, random_state=0).fit_transform(X_data)
    k_mean=KMeans(n_clusters=1,random_state=0).fit(x_dr)
    global_centers=k_mean.cluster_centers_
    plt.figure()  # 创建一个画布
    plt.scatter(global_centers[0, 0], global_centers[0, 1], s=50,marker='o', color='r')
    #global_centers=global_centers*
    if isfold and plot_list is not None:
        color_temp=['orange','darkorange','blue','b','g','green']
        for i in range(data_size):
            num_sample=(plot_list[i+1]-plot_list[i])//2

            # 计算EMD
            data_original_flat = np.mean(x_dr[plot_list[i]:plot_list[i]+num_sample], axis=1)
            data_generated_flat = np.mean(x_dr[plot_list[i]:plot_list[i+1]], axis=1)

            emd = wasserstein_distance(data_original_flat, data_generated_flat)
            print(f"Earth Mover's Distance (EMD): {emd}")
            k_mean = KMeans(n_clusters=1,random_state=0).fit(x_dr[plot_list[i]:plot_list[i]+num_sample])
            cluster_centers_ = k_mean.cluster_centers_
            plt.scatter(cluster_centers_[0,0],cluster_centers_[0,1],s=50,marker= 'o', edgecolors='k',color=color_temp[2*i],linewidths=3)
            #plt.plot(cluster_centers_[0],marker= 'o', color=color_temp[2*i])
            # plt.plot(cluster_centers_[0], global_centers[0], color = 'black')

            plt.scatter(x_dr[plot_list[i]:plot_list[i]+num_sample, 0], x_dr[plot_list[i]:plot_list[i]+num_sample, 1],marker='.',s=30,color=color_temp[2*i],label=i)
            plt.scatter(x_dr[plot_list[i] + num_sample:plot_list[i+1], 0],x_dr[plot_list[i] + num_sample:plot_list[i+1], 1], marker='^',s=20,color=color_temp[2*i+1],label=i)

            k_mean = KMeans(n_clusters=1,random_state=0).fit(x_dr[plot_list[i]:plot_list[i+1]])
            cluster_centers_ = k_mean.cluster_centers_
            plt.scatter(cluster_centers_[0, 0], cluster_centers_[0, 1], s=50, marker='o',edgecolors='r', color=color_temp[2*i+1],linewidths=2)
            # plt.plot(cluster_centers_[0], global_centers[0], color = 'brown')

    else:
        for i in range(data_size):
            plt.scatter(x_dr[Y_data == i, 0], x_dr[Y_data == i, 1], label=i)  # ,c=color([i]))
    plt.legend()  # 显示图例

    if isdata:
        plt.title("PCA Distribution {}- Hospital{}".format(Dataset_name,ith))
    elif isfold:
        plt.title("PCA of {} - fold{}".format(Dataset_name,ith))  # 显示标题
    else:
        plt.title("PCA of {} Cluster".format(Dataset_name))
    plt.show()

def View_Vectorize(data_list):
    view_vec = []
    view_data = data_list.transpose(0, 3, 2, 1)
    for data in view_data:
        view_vec.append(np.stack([data[i][np.triu_indices(data.shape[1], k=1)] for i in range(N_views)]))
    return np.array(view_vec).transpose(0, 2, 1)


def simulate_feature_std(mean_list, std_list, train_shample):
    simulate_feature = []
    for i in range(mean_list.shape[0]):
        mean = mean_list[i]
        std = std_list[i]
        mean_s = mean + (-1 + 2 * np.random.rand()) % (mean / 10)
        std_s = std + (-1 + 2 * np.random.rand()) % (std / 10)
        simulate_feature.append(np.abs(np.random.normal(mean_s, std_s, len(train_shample))))
    X = np.array(simulate_feature).transpose(1, 0)
    X[np.isnan(X)] = 0
    X = np.array(
        [[antiVectorize(i[j * (len(i) // N_views):(j + 1) * (len(i) // N_views)], N_Nodes) for j in range(N_views)] for
         i in X])
    return X.transpose(0, 3, 2, 1)


def meta_simulate_feature_cov11(mean_list, std_list, train_shample):
    random_mean = -max(mean_list) + 2 * max(mean_list) * np.random.rand(len(mean_list))
    random_std = -max(std_list) + 2 * max(std_list) * np.random.rand(len(std_list))
    mean_s = mean_list + np.nan_to_num()
    std_s = std_list + np.nan_to_num()
    X1 = np.abs(np.random.multivariate_normal(mean_s, std_s, len(train_shample)))
    X1 = np.array(
        [[antiVectorize(i[j * (len(i) // N_views):(j + 1) * (len(i) // N_views)], N_Nodes) for j in range(N_views)]
         for i in X1]).transpose(0, 3, 2, 1)
    return X1


def simulate_feature_cov(mean_list, std_list, random_num1, random_num2, train_shample):
    simulate_feature = []
    bais_mean = abs(random_num1) % abs(mean_list / 10)
    if random_num1 < 0:
        bais_mean = -bais_mean
    bais_std = abs(random_num2) % abs(std_list / 10)
    if random_num2 < 0:
        bais_std = -bais_std
    mean_s = mean_list + np.nan_to_num(bais_mean)
    std_s = std_list + np.nan_to_num(bais_std)
    X1 = np.abs(np.random.multivariate_normal(mean_s, std_s, len(train_shample)))
    X1 = np.array(
        [[antiVectorize(i[j * (len(i) // N_views):(j + 1) * (len(i) // N_views)], N_Nodes) for j in range(N_views)]
         for i in X1]).transpose(0, 3, 2, 1)
    return X1


def meta_simulate_feature_cov1(mean_list, std_list, random_num1, random_num2, train_shample, i):
    simulate_feature = []
    bais_mean = abs(random_num1) % abs(mean_list / i)
    if random_num1 < 0:
        bais_mean = -bais_mean
    bais_std = abs(random_num2) % abs(std_list / i)
    if random_num2 < 0:
        bais_std = -bais_std

    mean_s = mean_list + np.nan_to_num(bais_mean)
    std_s = std_list + np.nan_to_num(bais_std)
    X1 = np.abs(np.random.multivariate_normal(mean_s, std_s, len(train_shample)))
    X1 = np.array(
        [[antiVectorize(i[j * (len(i) // N_views):(j + 1) * (len(i) // N_views)], N_Nodes) for j in range(N_views)]
         for i in X1]).transpose(0, 3, 2, 1)
    return X1


def simulate_feature_cov4(mean_list, std_list, train_shample):
    simulate_feature = []
    vv = np.random.rand() % (mean_list / 10)
    n = -1 + 2 * np.random.rand()

    mean_s = mean_list + np.nan_to_num((-1 + 2 * np.random.rand(len(mean_list))) % abs(mean_list / 5))
    std_s = std_list + np.nan_to_num((-1 + 2 * np.random.rand(len(std_list), len(std_list))) % abs(std_list / 5))

    X1 = np.abs(np.random.multivariate_normal(mean_s, std_s, len(train_shample)))
    X1 = np.array(
        [[antiVectorize(i[j * (len(i) // N_views):(j + 1) * (len(i) // N_views)], N_Nodes) for j in range(N_views)]
         for i in X1]).transpose(0, 3, 2, 1)
    return X1


def simulate_feature_cov3(mean_list, std_list, train_shample, i):
    simulate_feature = []
    vv = np.random.rand() % (mean_list / 10)
    n = -1 + 2 * np.random.rand()
    mean_s = mean_list + np.nan_to_num((-1 + 2 * np.random.rand(len(mean_list))) % abs(mean_list / i))
    std_s = std_list + np.nan_to_num((-1 + 2 * np.random.rand(len(std_list), len(std_list))) % abs(std_list / i))
    X1 = np.abs(np.random.multivariate_normal(mean_s, std_s, len(train_shample)))
    X1 = np.array(
        [[antiVectorize(i[j * (len(i) // N_views):(j + 1) * (len(i) // N_views)], N_Nodes) for j in range(N_views)]
         for i in X1]).transpose(0, 3, 2, 1)
    return X1


def save_weights(fold_num, dgn, rdgn, name, classname):
    torch.save(
        rdgn.state_dict(),
        os.path.join(
            f"fold_{fold_num}_classname_{classname}",
            f"rdgn_{name}_fold_{fold_num}_classname_{classname}.pt",
        ),
    )
    torch.save(
        dgn.state_dict(),
        os.path.join(
            f"fold_{fold_num}_classname_{classname}",
            f"dgn_{name}_fold_{fold_num}_classname_{classname}.pt",
        ),
    )

    print(f"Weights saved with name {name}\n")

def KL_error(cbt, target_data, six_views=False):
    """
        Calculate the KL_divergence between the CBT and test subjects (all views)
        Args:
            cbt: models output
            target_data: list of data objects
    """
    cbt_dist = cbt.sum(axis=1)
    cbt_probs = cbt_dist / cbt_dist.sum()

    views = torch.cat([data.con_mat for data in target_data], axis=2).permute((2, 1, 0))
    # View 1
    view1_mean = views[range(0, views.shape[0], 6 if six_views else 4)].mean(axis=0)
    view1_dist = view1_mean.sum(axis=1)
    view1_prob = view1_dist / view1_dist.sum()
    kl_1 = ((cbt_probs * torch.log2(cbt_probs / view1_prob)).sum().abs()) + (
        (view1_prob * torch.log2(view1_prob / cbt_probs)).sum().abs())

    # View 2
    view2_mean = views[range(1, views.shape[0], 6 if six_views else 4)].mean(axis=0)
    view2_dist = view2_mean.sum(axis=1)
    view2_prob = view2_dist / view2_dist.sum()
    kl_2 = ((cbt_probs * torch.log2(cbt_probs / view2_prob)).sum().abs()) + (
        (view2_prob * torch.log2(view2_prob / cbt_probs)).sum().abs())

    # View 3
    view3_mean = views[range(2, views.shape[0], 6 if six_views else 4)].mean(axis=0)
    view3_dist = view3_mean.sum(axis=1)
    view3_prob = view3_dist / view3_dist.sum()
    kl_3 = ((cbt_probs * torch.log2(cbt_probs / view3_prob)).sum().abs()) + (
        (view3_prob * torch.log2(view3_prob / cbt_probs)).sum().abs())

    # View 4
    view4_mean = views[range(3, views.shape[0], 6 if six_views else 4)].mean(axis=0)
    view4_dist = view4_mean.sum(axis=1)
    view4_prob = view4_dist / view4_dist.sum()
    kl_4 = ((cbt_probs * torch.log2(cbt_probs / view4_prob)).sum().abs()) + (
        (view4_prob * torch.log2(view4_prob / cbt_probs)).sum().abs())

    if six_views:
        # View 5
        view5_mean = views[range(4, views.shape[0], 6 if six_views else 4)].mean(axis=0)
        view5_dist = view5_mean.sum(axis=1)
        view5_prob = view5_dist / view5_dist.sum()
        kl_5 = ((cbt_probs * torch.log2(cbt_probs / view5_prob)).sum().abs()) + (
            (view5_prob * torch.log2(view5_prob / cbt_probs)).sum().abs())

        # View 6
        view6_mean = views[range(5, views.shape[0], 6 if six_views else 4)].mean(axis=0)
        view6_dist = view6_mean.sum(axis=1)
        view6_prob = view6_dist / view6_dist.sum()
        kl_6 = ((cbt_probs * torch.log2(cbt_probs / view6_prob)).sum().abs()) + (
            (view6_prob * torch.log2(view6_prob / cbt_probs)).sum().abs())
    else:
        kl_5, kl_6 = 0, 0
    return kl_1, kl_2, kl_3, kl_4, kl_5, kl_6

def meta_simulate_cbt(model, re_model, data):
    model.eval()
    re_model.eval()
    test_casted = [i.to(device) for i in cast_data(data)]
    avg_test_mae = 0
    hat = []
    for test_sample in test_casted:
        cbt = model(test_sample)  # N_ROI, N_ROI
        cbt = cbt.unsqueeze(0).unsqueeze(0)
        data_hat = re_model(cbt)  # 1, n_views, N_ROI, N_ROI
        data_hat = data_hat.squeeze().permute(1, 2, 0)
        mae = torch.abs(data_hat - test_sample.con_mat).mean()
        hat.append(data_hat.cpu())
        avg_test_mae += mae
    return torch.stack(hat).detach().numpy()

def plotTSNE1(X_data, Y_data, data_size, ith, isfold, isdata,plot_list=None):
    color = plt.get_cmap('PuOr', len(data_size))
    Y_data = np.array(Y_data)
    X_data = np.array(X_data) #np.concatenate([X_data[Y_data==i] for i in data_size])

    x_dr = TSNE(n_components=2, random_state=0).fit_transform(X_data)
    plt.figure()  # 创建一个画布
    for i in data_size:
        plt.scatter(x_dr[Y_data == i, 0], x_dr[Y_data == i, 1], label=i)  # ,c=color([i]))
    plt.legend()  # 显示图例

    if isdata:
        plt.title("TSNE Distribution {}- Hospital{}".format(Dataset_name,ith))
    elif isfold:
        plt.title("TSNE of {} - fold{}".format(Dataset_name,ith))  # 显示标题
    else:
        plt.title("TSNE of {} Cluster".format(Dataset_name))
    plt.show()

def plotPCA1(X_data, Y_data, data_size, ith, isfold, isdata,plot_list=None):
    #color = plt.get_cmap('PuOr', data_size)
    Y_data = np.array(Y_data)
    X_data = np.array(X_data)

    x_dr = PCA(n_components=2, random_state=0).fit_transform(X_data)
    k_mean=KMeans(n_clusters=1,random_state=0).fit(x_dr)
    global_centers=k_mean.cluster_centers_
    plt.figure()  # 创建一个画布
    plt.scatter(global_centers[0, 0], global_centers[0, 1], s=50,marker='o', color='r')
    #global_centers=global_centers*
    if isfold and plot_list is not None:
        color_temp=['orange','darkorange','blue','b','g','green']
        for i in range((data_size)):
            num_sample=(plot_list[i+1]-plot_list[i])//2
            k_mean = KMeans(n_clusters=1,random_state=0).fit(x_dr[plot_list[i]:plot_list[i]+num_sample])
            cluster_centers_ = k_mean.cluster_centers_
            plt.scatter(cluster_centers_[0,0],cluster_centers_[0,1],s=50,marker= 'o', edgecolors='k',color=color_temp[2*i],linewidths=3)
            #plt.plot(cluster_centers_[0],marker= 'o', color=color_temp[2*i])
            # plt.plot(cluster_centers_[0], global_centers[0], color = 'black')

            plt.scatter(x_dr[plot_list[i]:plot_list[i]+num_sample, 0], x_dr[plot_list[i]:plot_list[i]+num_sample, 1],marker='.',s=30,color=color_temp[2*i],label=i)
            plt.scatter(x_dr[plot_list[i] + num_sample:plot_list[i+1], 0],x_dr[plot_list[i] + num_sample:plot_list[i+1], 1], marker='^',s=20,color=color_temp[2*i+1],label=i)

            k_mean = KMeans(n_clusters=1,random_state=0).fit(x_dr[plot_list[i]:plot_list[i+1]])
            cluster_centers_ = k_mean.cluster_centers_
            plt.scatter(cluster_centers_[0, 0], cluster_centers_[0, 1], s=50, marker='o',edgecolors='r', color=color_temp[2*i+1],linewidths=2)
            # plt.plot(cluster_centers_[0], global_centers[0], color = 'brown')

    else:
        for i in data_size:
            plt.scatter(x_dr[Y_data == i, 0], x_dr[Y_data == i, 1], label=i)  # ,c=color([i]))
    plt.legend()  # 显示图例

    if isdata:
        plt.title("PCA Distribution {}- Hospital{}".format(Dataset_name,ith))
    elif isfold:
        plt.title("PCA of {} - fold{}".format(Dataset_name,ith))  # 显示标题
    else:
        plt.title("PCA of {} Cluster".format(Dataset_name))
    plt.show()


def find_min_len(vec,pred):
    min=len(vec)
    index=0
    for i in range(max(pred+1)):
        if min>len(vec[pred==i]):
            min=len(vec[pred==i])
            index=i
    return index