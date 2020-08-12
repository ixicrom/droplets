from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pl
from sklearn import manifold
from processing_tools import * # also imports all from data_tools and slice_tools
from sklearn.cluster import KMeans

filePath='/Users/s1101153/Desktop/droplet_stacks/63x/rect_pickles'
dat=read_files(filePath)
idx = pd.IndexSlice
colms=['val_green', 'val_red']
vals = dat.loc[:,idx[:,colms]].transpose()
vals
vals.shape

# vals.to_numpy()


scaler = StandardScaler()
scaled_vals=scaler.fit_transform(vals)
scaled_vals.shape
s_vals = pd.DataFrame(scaled_vals, index=vals.index)


names = s_vals.index.get_level_values(0)
sample = names.str[4:20].str.rstrip('_stack')
samples = sample.unique().values
samples
cmap = pl.get_cmap('tab20')
colours = cmap(np.linspace(start=0,stop=1,num=len(samples)))
len(samples)
len(colours)
clr = {samples[i]: colours[i] for i in range(len(samples))}


perp_vals = [5,10,30,50,80,100]

i=1
for p in perp_vals:
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=p)
    dat_tsne = tsne.fit_transform(s_vals)
    pl.subplot(2,3,i)
    for point in range(dat_tsne.shape[0]):
        pl.scatter(dat_tsne[point,0], dat_tsne[point,1], color=clr[sample[point]], label=sample[point])
    handles, labels = pl.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    pl.title('Perplexity = '+str(p))
    i=i+1
pl.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.0,1.0))
pl.show()




# %% clustering on t-SNE results

tsne_10 = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=10)
dat_tsne_10 = tsne_10.fit_transform(s_vals)
np.random.seed(1234)
km = KMeans(n_clusters=5)
kmeans_tSNE = km.fit(dat_tsne_10)
labs = pd.DataFrame(kmeans_tSNE.labels_, index=vals.index, columns=['cluster'])
labs

names = labs.index.get_level_values(0)
cols = labs.index.get_level_values(1)
cols=cols.str.lstrip('val_')
samples=names.str.lstrip('SUM_').str[0:18].str.rstrip('_stack')
slices = names.str[-7:].str.lstrip('_')
labs['colour'] = cols
labs['sample'] = samples
labs['slice'] = slices

count = count_clusters(labs.reset_index(), counter='cluster', grouper1='sample', grouper2='colour')

count.transpose().plot(kind='bar', stacked=True)
pl.legend(loc='center left', bbox_to_anchor=(1.0,0.5))

pl.xlabel('Original image')
pl.ylabel('Cluster frequency')
pl.title('K-means after t-SNE')
pl.show()

# %% gini score plots
np.mean(gini_score(count))
# reminder: a lower gini score is better. So this score of 0.2 is good!

scores5=list()
scores10=list()
scores30=list()
scores50=list()
scores80=list()
scores100=list()
scores=[scores5, scores10, scores30, scores50, scores80, scores100]
x = [*range(2,33)]

names = vals.index.get_level_values(0)
cols = vals.index.get_level_values(1)
cols=cols.str.lstrip('val_')
samples=names.str.lstrip('SUM_').str[0:18].str.rstrip('_stack')
slices = names.str[-7:].str.lstrip('_')


for i in range(len(perp_vals)):
    score=list()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=perp_vals[i])
    dat_tsne = tsne.fit_transform(s_vals)
    for n in x:
        np.random.seed(1234)
        km = KMeans(n_clusters = n)
        kmeans_fit=km.fit(dat_tsne)
        labs = pd.DataFrame(kmeans_fit.labels_, index=vals.index, columns=['cluster'])
        labs['colour'] = cols
        labs['sample'] = samples
        labs['slice'] = slices
        count = count_clusters(labs.reset_index(), counter='cluster', grouper1='sample', grouper2='colour')
        scores[i].append(np.mean(gini_score(count)))
    pl.plot(x, scores[i], label=str(perp_vals[i]))
pl.xlabel('Number of clusters')
pl.ylabel('Gini score')
pl.legend(title='t-SNE Perplexity')
pl.savefig('/Users/s1101153/Dropbox/Emily/Graphs/t-SNE_perplexity_clust-num_gini.png')
pl.show()


# I might also update the PCA_droplets.py gini scree plot to include t-SNE along with the other methods
