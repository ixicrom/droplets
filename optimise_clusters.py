from data_tools import *
from slice_tools import *
from sklearn import cluster
import random
import pandas as pd
import matplotlib.pyplot as pl
import math

def measure_clusters(cluster_freq):
    idx = pd.IndexSlice

    sample_scores = list()
    for col in cluster_freq.columns:
        counts = cluster_freq[col].values
        sumsq = np.sum(counts**2)
        n_clust = np.count_nonzero(counts)
        score = sumsq/n_clust
        sample_scores.append(score)
    return sample_scores

def gini_index(cluster_freq):
    sample_scores = list()
    n_clust = cluster_freq.shape[0]
    for col in cluster_freq.columns:
        counts = cluster_freq[col].values
        p = counts[counts!=0]/12
        q = 1-p
        G = sum(p*q)
        sample_scores.append(G/n_clust**(1/3))
    return sample_scores

def entropy(cluster_freq):
    sample_scores = list()
    n_clust = cluster_freq.shape[0]
    for col in cluster_freq.columns:
        counts = cluster_freq[col].values
        p = counts[counts!=0]/12
        q = np.log(p)
        D = -1*sum(p*q)
        sample_scores.append(D/n_clust**(1/3))
    return sample_scores


def main():
    filePath='/Users/s1101153/Dropbox/Emily/'
    dat=read_files(filePath)
    dat = norm_data(dat)
    dat_all = theta_average(dat)
    dat_arr = dat_all.transpose().to_numpy()
    N=1
    dat_r_bins = dat_all.groupby(np.arange(len(dat_all))//N).mean()
    dat_forLearning = dat_r_bins.transpose()

    cluster_scores_gini = list()
    cluster_scores_ent = list()
    cluster_scores = list()
    for n_clust in range(2,100):
        np.random.seed(1234)
        kmeans=cluster.KMeans(n_clusters=n_clust).fit(dat_forLearning)
        labs=kmeans.labels_
        k_clusters = pd.Series(labs, index = dat_forLearning.index, name='Cluster_kmeans')

        dat_results = pd.concat([dat_forLearning, k_clusters], axis=1)
        dat_results = dat_results.set_index('Cluster_kmeans', append = True)

        dat_toPlot = dat_results.stack().reset_index()
        dat_toPlot.columns = ['Slice', 'Colour', 'Cluster_kmeans', 'r', 'Value']
        dat_toPlot['Sample'] = dat_toPlot['Slice'].str.slice(0,8)
        dat_toPlot = dat_toPlot.drop(['r', 'Value'], axis='columns').drop_duplicates().reset_index(drop=True)
        cl_count = count_clusters(dat_toPlot, grouper1='Sample', grouper2='Colour', counter='Cluster_kmeans')
        # cl_count.transpose().plot(kind='bar', stacked = True, title=str(n_clust), legend=False)

        score_gini = sum(gini_index(cl_count))
        cluster_scores_gini.append(score_gini)

        score_ent = sum(entropy(cl_count))
        cluster_scores_ent.append(score_ent)

        score = sum(measure_clusters(cl_count))
        cluster_scores.append(score)
    pl.plot(cluster_scores)
    pl.plot(cluster_scores_gini)
    pl.plot(cluster_scores_ent)

main()

# below here is all testing, no need to actually run it.

def test_measure():
    mock_clusters = pd.DataFrame(index = range(50*12), columns = ['i', 'Cluster'])
    for i in range(50):
        for j in range(12):
            mock_clusters.iloc[(12*i)+j]['i'] = str(i)
            mock_clusters.iloc[(12*i)+j]['Cluster'] = random.randint(0,8)
    mock_clusters


    mock_clust_count = count_clusters(mock_clusters, grouper1='i', counter='Cluster')
    gini_scores = gini_index(mock_clust_count)
    ent_scores = entropy(mock_clust_count)
    gini_scores
    ent_scores

    ax = mock_clust_count.transpose().plot(kind='bar', stacked = True, legend=False)
    ax.set_xticklabels(scores_str)
    fig = ax.get_figure()
    fig.savefig('test.png')
    mock_clust_count
