import pandas as pd
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy


def norm_data(data, colms = ['val_green','val_red']):
    idx=pd.IndexSlice
    x = data.loc[:,idx[:, colms]]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data.loc[:,idx[:,colms]] = x_scaled
    return data

def theta_average(data, rCol = []):
    # remove a dimension by averaging over theta
    idx=pd.IndexSlice
    if rCol == []:
        rCol = data.columns[0]
    data_means=data.groupby([rCol]).mean() #group by r for one of the slices, doesn't matter which one as they are all the same

    # take the green pixel values
    data_mean_green=data_means.loc[:,idx[:,'val_green']]
    data_mean_green.index.name='r'

    # take the red pixel values
    data_mean_red = data_means.loc[:,idx[:,'val_red']]
    data_mean_red.index.name='r'

    # combine into a final dataframe
    data_mean_all = pd.concat([data_mean_green, data_mean_red], axis=1)

    return data_mean_all


def h_cluster(dat_forLearning, cut_num, showPlot = True):
    Z = linkage(dat_forLearning, method='ward', optimal_ordering=True)
    if showPlot:
        mydendro = dendrogram(Z, labels=dat_forLearning.index, truncate_mode='lastp')
        plt.show()
    Z_cut = hierarchy.cut_tree(Z, n_clusters = cut_num)

    # get the leaf labels out and make df for output
    Z_leaves = hierarchy.leaves_list(Z)
    Z_leaves=dat_forLearning.index[Z_leaves]
    Z_results = pd.DataFrame([Z_leaves, Z_cut]).transpose()

    images = Z_results[0].values
    slices = [x[0] for x in images]
    colours = [x[1] for x in images]
    Z_clusters = Z_results[1].values
    clusters_h = [x[0] for x in Z_clusters]
    Z_results[1] = clusters_h
    Z_results.index = [slices, colours]
    Z_results.columns = ['Image','Cluster_hier']
    Z_results = Z_results.drop(columns='Image')
    return Z_results
