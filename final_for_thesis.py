from full_analysis_tools import read_files, format_rectangles, read_calc_format_wedges, hier, clust, gini_score_range, optimalK, tSNE_transform, gini_score, phi_plot, PCA_transform
import matplotlib.pyplot as pl
import os
import numpy as np

graph_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP/Graphs/final_for_thesis/'

# %% rectangular/theta-averaged data
filePath = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/rect_pickles'
imagePath = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/final_images/ims_to_read/'

dat = read_files(filePath)
file_suffix = ''
if input('Use theta-averaged data? y/n: ') == 'y':
    r_dat = format_rectangles(dat, scale='standard', theta_av=True)
else:
    r_dat = format_rectangles(dat, scale='standard', theta_av=False)
    file_suffix += '_rect'
# %% calculate wedge data
wedge_path = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/'

data_tag = input('Which data to include? all/nice/oneA/oneB: ')
if data_tag == 'all':
    info_file = os.path.join(wedge_path, 'stack_info.csv')
    save_file = os.path.join(wedge_path, 'wedges_all')
    file_suffix += '_all'
elif data_tag == 'nice':
    info_file = os.path.join(wedge_path, 'stack_info_2020-08-20.csv')
    save_file = os.path.join(wedge_path, 'wedges_nice')
    file_suffix += 'nice'
elif data_tag == 'oneA':
    info_file = os.path.join(wedge_path, 'stack_info_2020-08-28_A.csv')
    save_file = os.path.join(wedge_path, 'wedges_A')
    file_suffix += 'oneA'
elif data_tag == 'oneB':
    info_file_one_B = os.path.join(wedge_path, 'stack_info_2020-08-28_B.csv')
    save_file_B = os.path.join(wedge_path, 'wedges_B')
    file_suffix += 'oneB'
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
r_dat = r_dat[r_dat.index.isin(wedges.index)]

# %% k-means/hier theta-averaged: gini score
if input('Use PCA? y/n: ') == 'y':
    file_suffix += '_pca'
    r_dat = PCA_transform(r_dat, 0.97)[1]

if input('Make gini score graph? y/n: ') == 'y':
    scores_k = gini_score_range(method='k',
                                dat=r_dat,
                                start_num=3,
                                stop_num=33)
    scores_h = gini_score_range(method='h',
                                dat=r_dat,
                                start_num=3,
                                stop_num=33)
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
    pl.savefig(graph_folder+'gap_'+file_suffix+'.png')
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
pl.savefig(graph_folder+'kmeans_bar_'+file_suffix+'.png')
pl.show()

k_gini = gini_score(k_count)
print('kmeans_gini'+file_suffix+': '+str(np.mean(k_gini)))

h_labs, h_count = hier(r_dat, 5, show_dendro=False)
h_gini = gini_score(h_count)
print('hier_gini'+file_suffix+': '+str(np.mean(h_gini)))

h_count.transpose().plot(kind='bar', stacked=True)
pl.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
pl.xlabel('Original image')
pl.ylabel('Cluster frequency')
pl.title('Hierarchical clusters '+file_suffix[1:])
pl.tight_layout()
pl.savefig(graph_folder+'hier_bar_'+file_suffix+'.png')
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

# %% removed bad images ________________________________


# %% one image per composition______________________________________
