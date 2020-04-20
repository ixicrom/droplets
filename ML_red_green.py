


from slice_tools import *
from sklearn import cluster
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


# def plot_dendrogram(model, **kwargs):
#     # Create linkage matrix and then plot the dendrogram
#
#     # create the counts of samples under each node
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for i, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[i] = current_count
#
#     linkage_matrix = np.column_stack([model.children_, model.distances_,
#                                       counts]).astype(float)
#
#     # Plot the corresponding dendrogram
#     dendrogram(linkage_matrix, **kwargs)


def main():
    # read in files and drop NAs
    filePath='/Users/s1101153/Dropbox/Emily/'
    dat=read_files(filePath)
    dat = dat.dropna()

    # remove a dimension by averaging over theta
    dat_means=dat.groupby([('T6M_29_1_slice5.pkl','r')]).mean() #group by r for one of the slices, doesn't matter which one as they are all the same
    idx=pd.IndexSlice

    # take the green pixel values
    dat_mean_green=dat_means.loc[:,idx[:,'val_green']]
    dat_mean_green.index.name='r'
    dat_mean_green

    dat_mean_red = dat_means.loc[:,idx[:,'val_red']]
    dat_mean_red.index.name='r'
    dat_mean_red

    dat_all = pd.concat([dat_mean_green, dat_mean_red], axis=1)
    dat_all.transpose().index
    dat_arr = dat_all.transpose().to_numpy()
    dat_arr.shape
    dat_final = dat_all.transpose()

    np.random.seed(1234)
    # kmeans=cluster.KMeans(n_clusters=9).fit(dat_arr)
    # pred = kmeans.predict(dat_final
    # pred
    # dat_final['Cluster'] = pd.Series(pred, index = dat_final.index)
    # dat_print=dat_final['Cluster'].sort_index()
    # labs=kmeans.labels_
    # labs
    # len(labs)
    # np.savetxt('test_2020-04-14.txt',np.asarray(labs))
    # dat_print.to_csv('test_2020-04-17.csv', header=False)

    # h_cluster = cluster.AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(dat_arr)

    # plt.title('Hierarchical Clustering Dendrogram')
    # plot_dendrogram(h_cluster)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()



    Z = linkage(dat_final, method='ward', optimal_ordering=False)
    Z
    dendrogram(Z, labels=dat_final.index)
    plt.show()
main()
