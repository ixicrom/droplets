from slice_tools import *
from sklearn import cluster, preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import math
from datetime import datetime


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
    norm = False
    filePath='/Users/s1101153/Dropbox/Emily/'
    dat=read_files(filePath)
    dat = dat.dropna()
    idx=pd.IndexSlice

# rescale the data______________________________________________
    if(input('Normalise the data? (y/n): ')=='y'):
        norm = True
        x = dat.loc[:,idx[:,['val_green','val_red']]]
        x.shape
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        # pd.DataFrame(x_scaled).describe()
        dat.loc[:,idx[:,['val_green','val_red']]] = x_scaled
    dat
# ____________________________________________________________


    plt.scatter(x,y,c=valr, cmap='Reds')
    plt.xlabel("r (pixels)")
    plt.ylabel("theta (rad)")
    plt.title("Slice 3")
    plt.savefig("T2M_6_1_slice3_norm_r.png")
    plt.close()


    # remove a dimension by averaging over theta
    dat_means=dat.groupby([('T6M_29_1_slice5.pkl','r')]).mean() #group by r for one of the slices, doesn't matter which one as they are all the same
    # dat_means

    # take the green pixel values
    dat_mean_green=dat_means.loc[:,idx[:,'val_green']]
    dat_mean_green.index.name='r'
    # dat_mean_green

    dat_mean_red = dat_means.loc[:,idx[:,'val_red']]
    dat_mean_red.index.name='r'
    # dat_mean_red

    dat_all = pd.concat([dat_mean_green, dat_mean_red], axis=1)
    # dat_all.transpose().index
    dat_arr = dat_all.transpose().to_numpy()
    dat_arr.shape


# average over groups of 20 r values to bin data
    N = int(input('How many r values to average over: '))
    print(N)
    dat_r_bins = dat_all.groupby(np.arange(len(dat_all))//N).mean()
    dat_r_bins
    dat_forLearning = dat_r_bins.transpose()
    # dat_forLearning

# k-means clustering___________________________________________
    np.random.seed(1234)
    kmeans=cluster.KMeans(n_clusters=9).fit(dat_forLearning)
    # pred = kmeans.predict(dat_forLearning)
    # pred
    labs=kmeans.labels_
    # labs
    len(labs)

    k_clusters = pd.Series(labs, index = dat_forLearning.index, name='Cluster_kmeans')
    # k_clusters

# ____________________________________________________________

# hierarchical clustering and plotting with sklearn (doesn't currently work)
    # np.random.seed(1234)
    # h_cluster = cluster.AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(dat_arr)
    #
    # plt.title('Hierarchical Clustering Dendrogram')
    # plot_dendrogram(h_cluster)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()
# ____________________________________________________________

# working hierarchical clustering with skcipy linkage, and plot____
    np.random.seed(1234)
    Z = linkage(dat_forLearning, method='ward', optimal_ordering=True)
    mydendro = dendrogram(Z, labels=dat_forLearning.index, truncate_mode='lastp')
    plt.show()
    cut_num = int(input("Number of clusters to cut into (if in doubt, choose 9): "))
    # Z_tree = hierarchy.to_tree(Z)
    Z_cut = hierarchy.cut_tree(Z, n_clusters = cut_num)
    # Z_cut
    # get the leaf labels out and print to a file
    Z_leaves = hierarchy.leaves_list(Z)
    Z_leaves=dat_forLearning.index[Z_leaves]
    Z_results = pd.DataFrame([Z_leaves, Z_cut]).transpose()
    # Z_results

    images = Z_results[0].values
    slices = [x[0] for x in images]
    colours = [x[1] for x in images]
    Z_clusters = Z_results[1].values
    clusters_h = [x[0] for x in Z_clusters]
    Z_results[1] = clusters_h
    Z_results.index = [slices, colours]
    Z_results.columns = ['Image','Cluster_hier']
    # Z_results
    Z_results = Z_results.drop(columns='Image')
    # Z_results
    # Z_leaves.to_frame().to_csv('/Users/s1101153/Dropbox/Emily/Graphs/2020-04-29_leaves_av10r.csv')

# _________________________________________________________________


# data manipulation for output
    dat_results = pd.concat([dat_forLearning, k_clusters, Z_results], axis=1)
    dat_results = dat_results.set_index('Cluster_kmeans', append = True)
    dat_results = dat_results.set_index('Cluster_hier', append=True)
    # dat_results
    dat_toPlot = dat_results.stack().reset_index()
    dat_toPlot.columns = ['Slice', 'Colour', 'Cluster_kmeans', 'Cluster_hier', 'r', 'Value']
    dat_toPlot['Sample'] = dat_toPlot['Slice'].str.slice(0,8)
    dat_toPlot

    date = datetime.today().strftime('%Y-%m-%d')
    if norm:
        fileName = date + '_k_h_av' + str(N) + 'r' + str(cut_num) + 'h-clusters.csv'
    else:
        fileName = date + '_un-norm_k_h_av' + str(N) +'r_' + str(cut_num) + 'h-clusters.csv'
    filePath = '/Users/s1101153/Dropbox/Emily/Results/'
    outFile = filePath+fileName
    dat_toPlot.to_csv(outFile, header=True)
    print('File saved: '+outFile)
main()
