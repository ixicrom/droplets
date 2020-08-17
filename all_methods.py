from processing_tools import *
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import manifold
from collections import Counter


# %% dealing with the rectangular slices to make dataframe of r values

filePath='/Users/s1101153/Desktop/droplet_stacks/63x/rect_pickles'
dat=read_files(filePath)
idx=pd.IndexSlice


# %% scale data for PCA and t-SNE application
scaler = StandardScaler()
colms=['val_green', 'val_red']
vals = dat.loc[:,idx[:,colms]].transpose()
vals_scaled = scaler.fit_transform(vals)
dat_full_scaled = pd.DataFrame(vals_scaled, index=vals.index)

dat_full_scaled = dat_full_scaled.reset_index()
samples=dat_full_scaled['slices'].str.slice(4,21).str.rstrip('_stack')

dat_full_scaled.insert(0,'sample', samples)
slice_nums=dat_full_scaled['slices'].str.rstrip('.pkl').str[-2:].str.lstrip('e').astype(int)
slice_nums
dat_full_scaled['slices']=slice_nums
dat_full_scaled.rename(columns={'slices':'slice'}, inplace=True)

colours = dat_full_scaled['vars'].str[4:]

dat_full_scaled.insert(1,'colour', colours)
dat_full_scaled = dat_full_scaled.drop('vars', axis=1)

dat_full_scaled=dat_full_scaled.set_index(['sample', 'colour', 'slice'])
dat_full_scaled.shape
# %% min-max scaling for hierarchical and plain k-means clustering
dat_norm = norm_data(dat)

# remove a dimension by averaging over theta, for direct clustering
dat_all = theta_average(dat_norm)

dat_all = dat_all.transpose().reset_index()

samples=dat_all['slices'].str.slice(4,21).str.rstrip('_stack')

dat_all.insert(0,'sample', samples)
slice_nums=dat_all['slices'].str.rstrip('.pkl').str[-2:].str.lstrip('e').astype(int)
slice_nums
dat_all['slices']=slice_nums
dat_all.rename(columns={'slices':'slice'}, inplace=True)

colours = dat_all['vars'].str[4:]

dat_all.insert(1,'colour', colours)
dat_all = dat_all.drop('vars', axis=1)

dat_all=dat_all.set_index(['sample', 'colour', 'slice'])
# dat_all

# %% hierarchical clustering: 5 clusters

np.random.seed(1234)
hier = h_cluster(dat_all, 5, showPlot=True)
hier.index = dat_all.index
hier_count = count_clusters(hier.reset_index(), counter='Cluster_hier', grouper1='sample', grouper2 = 'colour')

hier_gini = gini_score(hier_count)

# %% kmeans clustering: 5 clusters

# scale theta-averaged features
dat_scaled=scaler.fit_transform(dat_all)

# k-means alone (on unscaled data)
np.random.seed(1234)
km = cluster.KMeans(n_clusters=5)
kmeans_normal = km.fit(dat_scaled)
labs_normal = pd.DataFrame(kmeans_normal.labels_, index=dat_all.index, columns=['normal'])

# PCA (on full scaled data without theta-averaging)
pca = PCA(.95)
dat_PCA = pca.fit_transform(dat_full_scaled.values)

# kmeans on PCA
np.random.seed(1234)
kmeans_PCA = km.fit(dat_PCA)
labs_PCA = pd.DataFrame(kmeans_PCA.labels_, index=dat_full_scaled.index, columns=['PCA'])

labs = pd.merge(labs_normal, labs_PCA, left_index=True, right_index=True)

# count and calculate gini scores
normal_count = count_clusters(labs.reset_index(), counter='normal', grouper1='sample', grouper2='colour')
PCA_count = count_clusters(labs.reset_index(), counter='PCA', grouper1='sample', grouper2='colour')
PCA_count
kmeans_gini = gini_score(normal_count)
PCA_kmeans_gini = gini_score(PCA_count)
PCA_kmeans_gini


# %% plotting

sample_names = hier_count.transpose().index.get_level_values(1)

gini_scores = pd.DataFrame()
gini_scores['sample'] = sample_names
gini_scores['hier'] = hier_gini
gini_scores['kmeans'] = kmeans_gini
gini_scores['PCA_kmeans'] = PCA_kmeans_gini
gini_scores
gini_scores.plot(x='sample', kind='bar')
pl.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
pl.xlabel('Original image')
pl.ylabel('Gini score')
pl.show()

# %% t-SNE: apply to scaled full data (not theta averaged)

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=10)
dat_tsne = tsne.fit_transform(dat_full_scaled)

# %% t-SNE plotting
pl.scatter(dat_tsne[:,0], dat_tsne[:,1])
pl.title('Perplexity = 10')
pl.show()
dat_full_scaled

# %% as before, colour based on original image
samples_full = dat_full_scaled.index.get_level_values(0)
samples_unique = samples_full.unique().values
cmap = pl.get_cmap('tab20')
colours = cmap(np.linspace(start=0,stop=1,num=len(samples_unique)))
len(samples_unique)
len(colours)
clr = {samples_unique[i]: colours[i] for i in range(len(samples_unique))}

for point in range(dat_tsne.shape[0]):
    pl.scatter(dat_tsne[point,0], dat_tsne[point,1],
        color=clr[samples_full[point]],
        label=samples_full[point],
        alpha=0.7)
handles, labels = pl.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
pl.title('Original image plotted on t-SNE components (p=10)')
pl.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.0,0.5))
pl.show()

# %% colour based on phir

phir_full = samples_full.str[6:15]
phir_full = phir_full.str.lstrip("5_").str.rstrip("_2")
phir_unique = phir_full.unique().values

colours_phir = cmap(np.linspace(start=0, stop=1, num=len(phir_unique)))
clr_phir = {phir_unique[i]: colours_phir[i] for i in range(len(phir_unique))}

for point in range(dat_tsne.shape[0]):
    pl.scatter(dat_tsne[point,0], dat_tsne[point,1],
        color=clr_phir[phir_full[point]],
        label=phir_full[point],
        alpha=0.7)
handles, labels = pl.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
pl.title('Phi_r value plotted on t-SNE components (p=10)')
pl.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.0,0.5))
pl.show()


# %% colour based on phip

phip_full = samples_full.str[0:7].str.rstrip('_phi')
phip_unique = phip_full.unique().values

colours_phip = cmap(np.linspace(start=0, stop=1, num=len(phip_unique)))
clr_phip = {phip_unique[i]: colours_phip[i] for i in range(len(phip_unique))}

for point in range(dat_tsne.shape[0]):
    pl.scatter(dat_tsne[point,0], dat_tsne[point,1],
        color=clr_phip[phip_full[point]],
        label=phip_full[point],
        alpha=0.7)
handles, labels = pl.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
pl.title('Phi_p value plotted on t-SNE components (p=10)')
pl.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.0,1.0))
pl.show()



# %% colour based on k-means cluster
cmap5 = pl.get_cmap('tab10')

n_clusters = 5
df_tsne = pd.DataFrame(dat_tsne, index=dat_full_scaled.index)
tsne_labels = pd.merge(df_tsne, labs, left_index=True, right_index=True)
tsne_km = tsne_labels['normal']
colours_km = cmap5(np.linspace(start=0, stop=0.5, num=n_clusters))

clr_km = {i: colours_km[i] for i in range(n_clusters)}
clr_km

for point in range(dat_tsne.shape[0]):
    pl.scatter(tsne_labels[0][point], tsne_labels[1][point],
        color=clr_km[tsne_km[point]],
        label=tsne_km[point],
        alpha=0.6)
handles, labels = pl.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
pl.title('K-means cluster plotted on t-SNE components (p=10)')
pl.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.0,0.5), title='Raw k-means cluster')
pl.show()


# %% colour based on hierarchical cluster

tsne_labels_hier = pd.merge(tsne_labels, hier, left_index=True, right_index=True)
tsne_labels_hier
tsne_hier = tsne_labels_hier['Cluster_hier']
for point in range(dat_tsne.shape[0]):
    pl.scatter(tsne_labels_hier[0][point], tsne_labels_hier[1][point],
        color=clr_km[tsne_hier[point]],
        label=tsne_hier[point],
        alpha=0.6)
handles, labels = pl.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
pl.title('Hierarchical cluster plotted on t-SNE components (p=10)')
pl.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.0,0.5), title='Hierarchical cluster')
pl.show()

# %% clusters on phir vs phip

all_clusters = tsne_labels_hier.drop([0, 1], axis=1)
all_clusters
all_clusters.insert(0, 'phir', phir_full.str[4:])
all_clusters.insert(0, 'phip', phip_full.str[4:].str.replace('-', '.'))

# pl.plot(all_clusters['phip'], all_clusters['phir'], linestyle='None', marker='x', alpha=0.5, c=all_clusters['normal'])
points = zip(all_clusters['phip'], all_clusters['phir'], all_clusters['normal'])
c=Counter(points)

for point in c.keys():
    pl.scatter(point[0], point[1],
        s=(c[point]**3),
        edgecolors=clr_km[point[2]],
        facecolors='none',
        label=point[2])
handles, labels = pl.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
pl.title('K-means clusters plotted on phi_r vs phi_p\n(size indicates number of points in that position)')
pl.xlabel('phi_p')
pl.ylabel('phi_r')
leg = pl.legend(by_label.values(), by_label.keys(), loc='center left',
                bbox_to_anchor=(1.0,0.5), title='K-means cluster')
for i in range(5):
    leg.legendHandles[i]._sizes=[30]
pl.show()
