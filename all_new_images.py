from full_analysis_tools import *
import pandas as pd


# %% rectangular data
filePath = '/Users/s1101153/Desktop/droplet_stacks/63x/rect_pickles'
imagePath = '/Users/s1101153/Desktop/droplet_stacks/63x/final_images/ims_to_read/'

r_dat = read_format_rectangles(filePath=filePath, scale='standard', theta_av=True)

rect_dat = read_format_rectangles(filePath=filePath, scale='standard', theta_av=False)


# %% wedge data
infoFileOld = '/Users/s1101153/Desktop/droplet_stacks/63x/stack_info.csv'

infoFile = '/Users/s1101153/Desktop/droplet_stacks/63x/stack_info_2020-08-20.csv'

readFile = '/Users/s1101153/Desktop/droplet_stacks/63x/nice_slices.pkl'
saveFile = '/Users/s1101153/Desktop/droplet_stacks/63x/nice_slices_2020-08-20.pkl'


old_wedges = read_calc_format_wedges(scale='minmax', fileName = readFile, reslice=False)
old_wedges

# old_wedges.reset_index()

new_wedges = read_calc_format_wedges(scale='minmax', fileName = saveFile, reslice = True, imPath = imagePath, infoFile = infoFile, hp=False)
new_wedges


# %% combined data
combi_dat = pd.concat([rect_dat, new_wedges], sort=False, axis=1)

combi_dat = combi_dat.dropna()

# %% rectangular data for just new images
new_rect_dat = rect_dat[rect_dat.index.isin(new_wedges.index)]
new_r_dat = r_dat[r_dat.index.isin(new_wedges.index)]








# %% apply PCA to new data and do k-means clustering

pca, dat_pca = PCA_transform(new_rect_dat, 0.95)

k_labs, k_count_pca = clust(method='k', n_clust=5, dat=dat_pca, col_name='kmeans_PCA')
k_count_pca
k_labs

# %% apply PCA to old data and do k-means clustering

pca, dat_pca_old = PCA_transform(rect_dat, 0.95)

k_labs_old, k_count_pca_old = clust(method='k', n_clust=5, dat=dat_pca, col_name='kmeans_PCA')
k_count_pca_old
k_labs_old


# %% plot post-PCA results for old and new data

score_old = gini_score_range(method='k', dat=dat_pca_old, start_num=3, stop_num=33)
score_new = gini_score_range(method='k', dat=dat_pca, start_num=3, stop_num=33)

r_score_old = gini_score_range(method='k', dat=r_dat, start_num=3, stop_num=33)
r_score_new = gini_score_range(method='k', dat=new_r_dat, start_num=3, stop_num=33)

x=[*range(3,33)]
pl.plot(x, score_old, label='All images, PCA')
pl.plot(x, score_new, label = 'Bad images removed, PCA')
pl.plot(x, r_score_old, label='All images, theta-average')
pl.plot(x, r_score_new, label='Bad images removed, theta-average')
pl.legend()
pl.xlabel('Number of clusters')
pl.ylabel('Gini score')
pl.title("K-means Gini Scores (standard scaled)")
pl.show()



# %%

k_count_pca_old.transpose().plot(kind='bar', stacked=True)
pl.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
pl.xlabel('Original image')
pl.ylabel('Cluster frequency')
pl.title('K-means standard scaled old')
pl.show()

k_count_pca.transpose().plot(kind='bar', stacked=True)
pl.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
pl.xlabel('Original image')
pl.ylabel('Cluster frequency')
pl.title('K-means standard scaled new')
pl.show()







# %% t-SNE visualisation of new data

dat_tsne_old = tSNE_transform(rect_dat, 10)
dat_tsne = tSNE_transform(new_rect_dat, 10)

dat_tsne_old.shape

pl.scatter(dat_tsne_old[:,0], dat_tsne_old[:,1])
pl.title('t-SNE standard scaled old, p=10')
pl.show()


# %% trying different perplexity values

perp_vals = [5,10,30,50,100]

for p in perp_vals:
    dat_tsne_test = tSNE_transform(new_rect_dat, p)
    pl.scatter(dat_tsne_test[:,0], dat_tsne_test[:,1])
    pl.title('t-SNE standard scaled new, p = ' +str(p))
    pl.show()


# %% try a different random seed

for p in perp_vals:
    dat_tsne_test = tSNE_transform(new_rect_dat, p, r_state=10)
    pl.scatter(dat_tsne_test[:,0], dat_tsne_test[:,1])
    pl.title('t-SNE standard scaled new, random_state=10, p=' +str(p))
    pl.show()


# %% look at which images we have in the p=5 case for new data

dat_tsne_new = tSNE_transform(new_rect_dat, 5)
sample_names=new_rect_dat.index.get_level_values(0)

tSNE_plot(dat_tsne_new, sample_names, 'New t-SNE, standard scaled, coloured by sample name', alpha=0.7)





# %% try k-means clustering on theta-averaged data

k_labs_old, k_count_old = kmeans(r_dat, 4, 'k-means theta-averaged old')
k_labs, k_count = kmeans(new_r_dat, 4, 'k-means theta-averaged')

phi_plot(k_labs_old, 'k-means theta-averaged old', 'K-means old, 4 clusters (theta-averaged, standard scaled)')
pl.show()

phi_plot(k_labs, 'k-means theta-averaged', 'K-means new, 4 clusters (theta-averaged, standard scaled)')
pl.show()


# %% try hierarchical clustering on theta-averaged data

h_labs_old, h_count_old = hier(r_dat, 4)
h_labs, h_count = hier(new_r_dat, 4)

phi_plot(h_labs_old, 'Cluster_hier', 'Hierarchical old, 4 clusters (theta-averaged, standard scaled)')
pl.show()

phi_plot(h_labs, 'Cluster_hier', 'Hierarchical new, 4 clusters (theta-averaged, standard scaled)')
pl.show()


# %%

k_labs_old_rect, k_count_old_rect = kmeans(rect_dat, 5, 'k-means old')
phi_plot(k_labs_old_rect, 'k-means old', 'K-means old rect standard scaled ')
pl.show()

# %%
h_labs_old_rect, h_count_old_rect = hier(rect_dat, 5)
phi_plot(h_labs_old_rect, 'Cluster_hier', 'Hierarchical old rect standard scaled ')
pl.show()
