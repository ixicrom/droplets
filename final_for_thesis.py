from full_analysis_tools import read_files, format_rectangles, read_calc_format_wedges, hier, clust, gini_score_range, optimalK, tSNE_transform, gini_score, phi_plot, PCA_transform
import matplotlib.pyplot as pl
import os
import numpy as np

graph_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP/Graphs/final_for_thesis/'

# %% rectangular/theta-averaged data
filePath = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/rect_pickles'
imagePath = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/final_images/ims_to_read/'

dat = read_files(filePath)
dat
r_dat = format_rectangles(dat, scale='standard', theta_av=True)

rect_dat = format_rectangles(dat, scale='standard', theta_av=False)
r_dat
# %% calculate wedge data
wedge_path = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/'

info_file_all = os.path.join(wedge_path, 'stack_info.csv')
save_file_all = os.path.join(wedge_path, 'wedges_all')

info_file_nice = os.path.join(wedge_path, 'stack_info_2020-08-20.csv')
save_file_nice = os.path.join(wedge_path, 'wedges_nice')

info_file_one_A = os.path.join(wedge_path, 'stack_info_2020-08-28_A.csv')
save_file_A = os.path.join(wedge_path, 'wedges_A')

info_file_one_B = os.path.join(wedge_path, 'stack_info_2020-08-28_B.csv')
save_file_B = os.path.join(wedge_path, 'wedges_B')

# wedges_all = read_calc_format_wedges(scale='minmax',
#                                      fileName=save_file_all,
#                                      reslice=True,
#                                      imPath=imagePath,
#                                      infoFile=info_file_nice,
#                                      hp=False)
#
# wedges_nice = read_calc_format_wedges(scale='minmax',
#                                       fileName=save_file_nice,
#                                       reslice=True,
#                                       imPath=imagePath,
#                                       infoFile=info_file_nice,
#                                       hp=False)
#
# wedges_A = read_calc_format_wedges(scale='minmax',
#                                    fileName=save_file_A,
#                                    reslice=True,
#                                    imPath=imagePath,
#                                    infoFile=info_file_one_A,
#                                    hp=False)
#
# wedges_B = read_calc_format_wedges(scale='minmax',
#                                    fileName=save_file_B,
#                                    reslice=True,
#                                    imPath=imagePath,
#                                    infoFile=info_file_one_B,
#                                    hp=False)
# wedges_all
# %% read wedge data (if previously calculated)
wedges_all = read_calc_format_wedges(scale='minmax',
                                     fileName=save_file_all,
                                     reslice=False,
                                     hp=False)

wedges_nice = read_calc_format_wedges(scale='minmax',
                                      fileName=save_file_nice,
                                      reslice=False,
                                      hp=False)

wedges_A = read_calc_format_wedges(scale='minmax',
                                   fileName=save_file_A,
                                   reslice=False,
                                   hp=False)

wedges_B = read_calc_format_wedges(scale='minmax',
                                   fileName=save_file_B,
                                   reslice=False,
                                   hp=False)


# %% working on all data_____________________________

# %% k-means/hier theta-averaged: gini score
# scores_k = gini_score_range(method='k',
#                            dat=r_dat,
#                            start_num=3,
#                            stop_num=33)
# scores_h = gini_score_range(method='h',
#                            dat=r_dat,
#                            start_num=3,
#                            stop_num=33)
# x = [*range(3, 33)]
# pl.plot(x, scores_k, label='K-means clustering')
# pl.plot(x, scores_h, label='Hierarchical clustering')
# pl.title('Gini score for all data')
# pl.xlabel('Number of clusters')
# pl.ylabel('Gini score')
# pl.legend(loc='lower right')
# pl.savefig(graph_folder+'gini_all.png')
# pl.show()

# %% k-means theta-averaged: gap statistic
# gap_count = optimalK(r_dat, maxClusters=30)
#
# pl.plot(gap_count[1]['clusterCount'], gap_count[1]['gap'])
# pl.xlabel('Number of clusters')
# pl.ylabel('Gap statistic')
# pl.title('Gap statistic for all data')
# pl.savefig(graph_folder+'gap_all.png')
# pl.show()

# %% t-SNE, try different perplexity values and two random seeds
# perp_vals = [5,10,30,50,100]
#
# for p in perp_vals:
#     dat_tsne_test = tSNE_transform(r_dat, p)
#     pl.scatter(dat_tsne_test[:,0], dat_tsne_test[:,1])
#     pl.title('t-SNE all data, p = ' +str(p))
#     pl.savefig(graph_folder+'tSNE_'+str(p)+'.png')
#     pl.show()
#
# for p in perp_vals:
#     dat_tsne_test = tSNE_transform(r_dat, p, r_state=10)
#     pl.scatter(dat_tsne_test[:,0], dat_tsne_test[:,1])
#     pl.title('t-SNE all data, random_state=10, p=' +str(p))
#     pl.savefig(graph_folder+'tSNE_r10_'+str(p)+'.png')
#     pl.show()

# %% k-means theta-averaged: 5 clusters
k_labs, k_count = clust(method='k',
                        n_clust=5,
                        dat=r_dat)

k_count.transpose().plot(kind='bar', stacked=True)
pl.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
pl.xlabel('Original image')
pl.ylabel('Cluster frequency')
pl.title('K-means clusters all data')
pl.savefig(graph_folder+'kmeans_bar_all.png')
pl.show()

k_gini = gini_score(k_count)
print(np.mean(k_gini))

# %% hier theta-averaged: 5 clusters
h_labs, h_count = hier(r_dat, 5, show_dendro=False)
h_gini = gini_score(h_count)
print(np.mean(h_gini))

h_count.transpose().plot(kind='bar', stacked=True)
pl.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
pl.xlabel('Original image')
pl.ylabel('Cluster frequency')
pl.title('Hierarchical clusters, all data')
pl.savefig(graph_folder+'hier_bar_all.png')
pl.show()

# %% hier and k-means phi plots

phi_plot(h_labs, 'Cluster_hier', 'Compositional plot of hierarchical cluster results')
phi_plot(k_labs, '', 'Compositional plot of k-means cluster results')

# %% PCA


# %% removed bad images ________________________________


# %% one image per composition______________________________________
