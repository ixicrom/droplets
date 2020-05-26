from data_tools import *
from slice_tools import *
from sklearn import cluster
import random

def main():
    filePath='/Users/s1101153/Dropbox/Emily/'
    dat=read_files(filePath)
    dat = norm_data(dat)
    dat_all = theta_average(dat)
    dat_arr = dat_all.transpose().to_numpy()
    N=1
    dat_r_bins = dat_all.groupby(np.arange(len(dat_all))//N).mean()
    dat_forLearning = dat_r_bins.transpose()

    for n_clust in range(2,20):
        n_clust = 8
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
        cl_count = count_clusters(dat_toPlot, 'Sample', 'Colour', 'Cluster_kmeans')
        cl_count.transpose().plot(kind='bar', stacked = True, title=str(n_clust), legend=False)

cl_count
dat_toPlot


def test_measure():
    mock_clusters = list()
    for i in range(50):
        mock_cluster = list()
        rand1 = random.randint(0, 12)
        mock_clusters.append(rand1)
        current_sum = rand1
        while current_sum<12:
            num_left = 12-current_sum
            new_rand = random.randint(0, num_left)
            mock_clusters.append(new_rand)
            current_sum = current_sum + new_rand
            # print(current_sum)
        # print(mock_cluster)
        mock_clusters.append(mock_cluster)
mock_clusters
