import pandas as pd
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import numpy as np

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
        plt.savefig('dendro.png')
        plt.show()
    Z_cut = hierarchy.cut_tree(Z, n_clusters = cut_num)

    # get the leaf labels out and make df for output
    Z_leaves = hierarchy.leaves_list(Z)
    Z_leaves=dat_forLearning.index[Z_leaves]
    Z_results = pd.DataFrame([Z_leaves, Z_cut]).transpose()

    images = Z_results[0].values
    samples = [x[0] for x in images]
    colours = [x[1] for x in images]
    slices = [x[2] for x in images]
    Z_clusters = Z_results[1].values
    clusters_h = [x[0] for x in Z_clusters]
    Z_results[1] = clusters_h
    Z_results.index = dat_forLearning.index
    Z_results.columns = ['Image','Cluster_hier']
    Z_results = Z_results.drop(columns='Image')
    return Z_results


def count_clusters(dat, counter, grouper1, grouper2="Null"):
    '''
    Takes in a dataframe dat
    For each combination of grouper1 and grouper2, makes a single entry
    Removes all columns apart from the combined entry and the counter column
    The counter column is the one that we want to know the frequency of
    For each entry, counts the frequency of the counter variable
    Returns a #entry x #counter dataframe, with frequency in each slot
    '''
    if grouper2 == "Null":
        dat['Entry'] = dat[grouper1]
    else:
        dat['Entry'] = dat[grouper1] + dat[grouper2]

    dat_small = dat[['Entry', counter]].copy()
    dat_small['MergedID'] = dat_small[counter].apply(str)+dat_small['Entry']
    counts = np.unique(dat_small['MergedID'], return_counts=True)
    dat_sorted = dat_small.sort_values(['MergedID']).drop_duplicates().reset_index(drop=True)
    dat_sorted = dat_sorted.drop('MergedID', axis='columns')
    dat_sorted['count'] = counts[1]

    cluster_count = dat_sorted.pivot(index=counter, columns='Entry').fillna(0)

    return(cluster_count)


def gini_score(cluster_freq):
    sample_scores = list() # make list for output scores

    for col in cluster_freq.columns: # will calculate a score for each image
        counts = cluster_freq[col].values
        p = counts[counts!=0]/sum(counts) # remove all zeroes (those clusters aren't present so don't want to include) and normalise
        q = 1-p
        G = sum(p*q) # calculate index
        sample_scores.append(G)
    return sample_scores
