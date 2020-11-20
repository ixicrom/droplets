from full_analysis_tools import read_files, format_rectangles, read_calc_format_wedges, hier, clust, gini_score_range, optimalK, tSNE_transform, gini_score, phi_plot, PCA_transform, tSNE_transform, tSNE_plot_2col
import matplotlib.pyplot as pl
import os
import numpy as np
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler

graph_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP/Graphs/final_for_thesis/'

# %% rectangular/theta-averaged data
filePath = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/rect_pickles'
imagePath = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/final_images/ims_to_read/'

dat = read_files(filePath)
file_suffix = ''
if input('Use theta-averaged data? y/n: ') == 'y':
    r_dat = format_rectangles(dat, scale='standard', theta_av=True)
    print(r_dat.head())
else:
    r_dat = format_rectangles(dat, scale='standard', theta_av=False)
    file_suffix += '_rect'
    print(r_dat.head())
# %% calculate wedge data
wedge_path = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/'

data_tag = input('Which data to include? all/nice/oneA/oneB: ')
if data_tag == 'all':
    info_file = os.path.join(wedge_path, 'stack_info.csv')
    save_file = os.path.join(wedge_path, 'wedges_all')+'.pkl'
    file_suffix += '_all'
elif data_tag == 'nice':
    info_file = os.path.join(wedge_path, 'stack_info_2020-08-20.csv')
    save_file = os.path.join(wedge_path, 'wedges_nice')+'.pkl'
    file_suffix += '_nice'
elif data_tag == 'oneA':
    info_file = os.path.join(wedge_path, 'stack_info_2020-08-28_A.csv')
    save_file = os.path.join(wedge_path, 'wedges_A')+'.pkl'
    file_suffix += '_oneA'
elif data_tag == 'oneB':
    info_file = os.path.join(wedge_path, 'stack_info_2020-08-28_B.csv')
    save_file = os.path.join(wedge_path, 'wedges_B')+'.pkl'
    file_suffix += '_oneB'
elif data_tag == 'noDist':
    info_file = os.path.join(wedge_path, 'stack_info_no_distorted.csv')
    save_file = os.path.join(wedge_path, 'wedges_noDist')+'.pkl'
    file_suffix += '_noDist'
elif data_tag == 'noIsh':
    info_file = os.path.join(wedge_path, 'stack_info_no_roundish.csv')
    save_file = os.path.join(wedge_path, 'wedges_noIsh')+'.pkl'
    file_suffix += '_noIsh'
else:
    print('Invalid selection, problems will happen')
if input('Calculate wedges from scratch? y/n: ') == 'y':
    wedges = read_calc_format_wedges(scale='minmax',
                                     fileName=save_file,
                                     reslice=True,
                                     imPath=imagePath,
                                     infoFile=info_file,
                                     hp=False)

else:
    wedges = read_calc_format_wedges(scale='minmax',
                                     fileName=save_file,
                                     reslice=False,
                                     hp=False)
if input('Include wedge data? y/n: ') == 'y':
    if input('Include only wedge data? y/n: ') == 'y':
        r_dat = wedges.dropna()
        r_arr = StandardScaler().fit_transform(r_dat)
        r_dat = pd.DataFrame(r_arr, index=r_dat.index, columns=r_dat.columns)
        file_suffix += '_wedge'
    else:
        r_dat = pd.concat([r_dat, wedges], sort=False, axis=1)
        r_arr = StandardScaler().fit_transform(r_dat)
        r_dat = pd.DataFrame(r_arr, index=r_dat.index, columns=r_dat.columns)
        r_dat = r_dat.dropna()
        file_suffix += '_combi'
else:
    r_dat = r_dat[r_dat.index.isin(wedges.index)]

# %% k-means/hier theta-averaged: gini score
if input('Use PCA? y/n: ') == 'y':
    var = float(input('PCA variance: '))
    file_suffix += '_pca'
    r_dat = PCA_transform(r_dat, var)[1]

if input('Make gini score graph? y/n: ') == 'y':
    start_t = time.time()
    scores_k = gini_score_range(method='k',
                                dat=r_dat,
                                start_num=3,
                                stop_num=33)
    scores_h = gini_score_range(method='h',
                                dat=r_dat,
                                start_num=3,
                                stop_num=33)
    stop_t = time.time()
    print('Gini plot time taken: '+str(stop_t-start_t))
    x = [*range(3, 33)]
    pl.plot(x, scores_k, label='K-means clustering')
    pl.plot(x, scores_h, label='Hierarchical clustering')
    pl.title('Gini score vs cluster number '+file_suffix[1:])
    pl.xlabel('Number of clusters')
    pl.ylabel('Gini score')
    pl.legend(loc='lower right')
    pl.savefig(graph_folder+'gini'+file_suffix+'.png')
    pl.show()

# %% k-means theta-averaged: gap statistic
if input('Make gap statistic graph? y/n: ') == 'y':
    gap_count = optimalK(r_dat, maxClusters=30)

    pl.plot(gap_count[1]['clusterCount'], gap_count[1]['gap'])
    pl.xlabel('Number of clusters')
    pl.ylabel('Gap statistic')
    pl.title('Gap statistic vs cluster number '+file_suffix[1:])
    pl.savefig(graph_folder+'gap'+file_suffix+'.png')
    pl.show()

# %% t-SNE, try different perplexity values and two random seeds
if input('Make t-SNE graphs? y/n: ') == 'y':
    perp_vals = [5, 10, 30, 50, 100]

    for p in perp_vals:
        dat_tsne_test = tSNE_transform(r_dat, p)
        pl.scatter(dat_tsne_test[:, 0], dat_tsne_test[:, 1])
        pl.title('t-SNE '+' data, p='+str(p)+' '+file_suffix[1:])
        pl.savefig(graph_folder+'tSNE_'+str(p)+file_suffix+'.png')
        pl.show()

    for p in perp_vals:
        dat_tsne_test = tSNE_transform(r_dat, p, r_state=10)
        pl.scatter(dat_tsne_test[:, 0], dat_tsne_test[:, 1])
        pl.title('t-SNE '+str(data_tag)+' data, random_state=10, p=' +str(p)+' '+file_suffix[1:])
        pl.savefig(graph_folder+'tSNE_r10_'+str(p)+file_suffix+'.png')
        pl.show()

# %% k-means theta-averaged: 5 clusters


k_labs, k_count = clust(method='k',
                        n_clust=5,
                        dat=r_dat)

k_count.transpose().plot(kind='bar', stacked=True)
pl.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
pl.xlabel('Original image')
pl.ylabel('Cluster frequency')
pl.title('K-means clusters '+file_suffix[1:])
pl.tight_layout()
pl.savefig(graph_folder+'kmeans_bar'+file_suffix+'.png')
pl.show()

k_gini = gini_score(k_count)
print('kmeans_gini'+file_suffix+': '+str(np.mean(k_gini)))

dendro = input('Show dendrogram? y/n: ') == 'y'
h_labs, h_count = hier(r_dat, 5, show_dendro=dendro)
h_gini = gini_score(h_count)
print('hier_gini'+file_suffix+': '+str(np.mean(h_gini)))

h_count.transpose().plot(kind='bar', stacked=True)
pl.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
pl.xlabel('Original image')
pl.ylabel('Cluster frequency')
pl.title('Hierarchical clusters '+file_suffix[1:])
pl.tight_layout()
pl.savefig(graph_folder+'hier_bar'+file_suffix+'.png')
pl.show()

# %% hier and k-means phi plots

pl.figure(num=None, figsize=(7, 8), dpi=80, facecolor='w', edgecolor='k')
phi_plot(h_labs,
         'Cluster_hier',
         'Compositional plot of hierarchical clustering '+file_suffix[1:],
         save_file=graph_folder+'hier_phi'+file_suffix+'.png')

pl.figure(num=None, figsize=(7, 8), dpi=80, facecolor='w', edgecolor='k')
phi_plot(k_labs,
         '',
         'Compositional plot of k-means clustering '+file_suffix[1:],
         save_file=graph_folder+'kmeans_phi'+file_suffix+'.png')


# %% t-SNE plots
if input('Plot clusters on t-SNE axes? y/n: ') == 'y':
    # k_labs['']
    dat_tsne = tSNE_transform(r_dat, 10)
    sample_names = r_dat.index.get_level_values(0)
    # sample_names
    # sample_names.unique()
    type(sample_names)
    clusters = pd.Series(k_labs[''])
    # clusters.unique()
    tSNE_plot_2col(dat_tsne, sample_names, clusters, 'tSNE test', alpha=0.7)


# def tSNE_plot_2col(dat_tsne, col_dat_1, col_dat_2, plot_title, **pl_kwargs):
#     '''
#     Plots col_dat on coordinates specified in dat_tsne.
#
#     Inputs:
#         dat_tsne: n x 2 DataFrame
#         col_dat: n x 1 series or index, in same order as dat_tsne
#         plot_title: string
#         **pl_kwargs: additional inputs for plotting with pl.scatter. Suggest setting alpha=0.7.
#
#     Outputs:
#         Displays a scatter plot of each coordinate in dat_tsne, with colour
#
#     '''
#     # colours for variable 1
#     if isinstance(col_dat_1, pd.core.indexes.base.Index):
#         col_unique_1 = col_dat_1.unique().values
#     else:
#         col_unique_1 = col_dat_1.unique()
#     n_colours_1 = len(col_unique_1)
#     if n_colours_1 > 10:
#         cmap_1 = pl.get_cmap('tab20')
#         plot_colours_1 = cmap_1(np.linspace(start=0, stop=1, num=n_colours_1))
#     elif n_colours_1 > 5:
#         cmap_1 = pl.get_cmap('tab10')
#         plot_colours_1 = cmap_1(np.linspace(start=0, stop=1, num=n_colours_1))
#     else:
#         cmap_1 = pl.get_cmap('tab10')
#         plot_colours_1 = cmap_1(np.linspace(start=0, stop=0.5, num=n_colours_1))
#
#     clr_1 = {col_unique_1[i]: plot_colours_1[i] for i in range(n_colours_1)}
#
#     # colours for variable 2
#     if isinstance(col_dat_2, pd.core.indexes.base.Index):
#         col_unique_2 = col_dat_2.unique().values
#     else:
#         col_unique_2 = col_dat_2.unique()
#     n_colours_2 = len(col_unique_2)
#     if n_colours_2 > 10:
#         cmap_2 = pl.get_cmap('tab20')
#         plot_colours_2 = cmap_2(np.linspace(start=0, stop=1, num=n_colours_2))
#     elif n_colours_2 > 5:
#         cmap_2 = pl.get_cmap('tab10')
#         plot_colours_2 = cmap_2(np.linspace(start=0, stop=1, num=n_colours_2))
#     else:
#         cmap_2 = pl.get_cmap('tab10')
#         plot_colours_2 = cmap_2(np.linspace(start=0, stop=0.5, num=n_colours_2))
#
#     clr_2 = {col_unique_2[i]: plot_colours_2[i] for i in range(n_colours_2)}
#
#     for point in range(dat_tsne.shape[0]):
#         pl.scatter(dat_tsne[point, 0], dat_tsne[point, 1],
#                    color=clr_1[col_dat_1[point]],
#                    label=col_dat_1[point],
#                    **pl_kwargs)
#         pl.scatter(dat_tsne[point, 0], dat_tsne[point, 1],
#                    color='none',
#                    edgecolor=clr_2[col_dat_2[point]],
#                    linewidths=2,
#                    label=col_dat_2[point],
#                    s=60,
#                    **pl_kwargs)
#     handles, labels = pl.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     pl.title(plot_title)
#     pl.legend(by_label.values(),
#               by_label.keys(),
#               loc='center left',
#               bbox_to_anchor=(1.0, 0.5))
#     pl.tight_layout()
#     pl.show()


# %% plots showing data input, copied lots from 2020-09-02_example_images.py

# first grab some example images
idx = pd.IndexSlice
dat25_g = r_dat.loc[idx['phip0-5_phir25', 'green'], :]
dat25_r = r_dat.loc[idx['phip0-5_phir25', 'red'], :]

dat75_g = r_dat.loc[idx['phip0-5_phir75', 'green'], :]
dat75_r = r_dat.loc[idx['phip0-5_phir75', 'red'], :]

dat40_g = r_dat.loc[idx['phip0-5_phir40', 'green'], :]
dat40_r = r_dat.loc[idx['phip0-5_phir40', 'red'], :]

# now get the cluster labels
labs25_r = h_labs.loc[idx['phip0-5_phir25', 'red'], :]
labs25_g = h_labs.loc[idx['phip0-5_phir25', 'green'], :]

labs75_r = h_labs.loc[idx['phip0-5_phir75', 'red'], :]
labs75_g = h_labs.loc[idx['phip0-5_phir75', 'green'], :]

labs40_r = h_labs.loc[idx['phip0-5_phir40', 'red'], :]
labs40_g = h_labs.loc[idx['phip0-5_phir40', 'green'], :]


# now see if the plot works!
fig, axs = pl.subplots(3, 2, sharex=True, sharey=True)
axs[0, 0].set_ylabel('slice')
axs[0, 0].imshow(dat25_r)
for i in range(dat25_r.shape[0]):
    for j in range(dat25_r.shape[1]):
        text = axs[0, 0].text(j, i, labs25_r.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[0, 0].set_title('phip=0.5, phir=25, red')

axs[1, 0].set_ylabel('slice')
axs[1, 0].imshow(dat25_g)
for i in range(dat25_g.shape[0]):
    for j in range(dat25_g.shape[1]):
        text = axs[1, 0].text(j, i, labs25_g.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[1, 0].set_title('phip=0.5, phir=25, green')

axs[0, 1].imshow(dat75_r)
for i in range(dat75_r.shape[0]):
    for j in range(dat75_r.shape[1]):
        text = axs[0, 1].text(j, i, labs75_r.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[0, 1].set_title('phip=0.5, phir=75, red')

axs[1, 1].imshow(dat75_g)
for i in range(dat75_g.shape[0]):
    for j in range(dat75_g.shape[1]):
        text = axs[1, 1].text(j, i, labs75_g.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[1, 1].set_title('phip=0.5, phir=75, green')

axs[2, 0].set_ylabel('slice')
axs[2, 0].set_xlabel('PCA component')
axs[2, 0].imshow(dat40_r)
for i in range(dat40_r.shape[0]):
    for j in range(dat40_r.shape[1]):
        text = axs[2, 0].text(j, i, labs40_r.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[2, 0].set_title('phip=0.5, phir=40, red')

axs[2, 1].set_xlabel('PCA component')
axs[2, 1].imshow(dat40_g)
for i in range(dat40_g.shape[0]):
    for j in range(dat40_g.shape[1]):
        text = axs[2, 1].text(j, i, labs40_g.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[2, 1].set_title('phip=0.5, phir=40, green')

fig.suptitle('Standard scaled theta-averaged data')
pl.show()
