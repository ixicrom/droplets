from processing_tools import *
from sklearn import cluster


imPath = '/Users/s1101153/Desktop/droplet_stacks/63x/final_images/ims_to_read/'
infoFile = '/Users/s1101153/Desktop/droplet_stacks/63x/stack_info.csv'

# %% calculating variables from wedge slices
slices = pd.read_pickle('/Users/s1101153/Desktop/droplet_stacks/63x/nice_slices.pkl')

slices_data = calc_variables(slices, highpass=False)

slices_data = slices_data.drop('imArray', axis=1)

slices_data
# %% dealing with the rectangular slices to make dataframe of r values

filePath='/Users/s1101153/Desktop/droplet_stacks/63x/rect_pickles'
dat=read_files(filePath)
idx=pd.IndexSlice

norm = True
dat = norm_data(dat)

# remove a dimension by averaging over theta
dat_all = theta_average(dat)

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
dat_all


# %% combining the two bits of data

full_data = pd.concat([slices_data, dat_all], sort=True, axis=1)

clean_data = full_data.dropna()
clean_data


# %%
np.random.seed(1234)
hier = h_cluster(dat_all, 50, showPlot=True)
hier.index = dat_all.index

gini_scores = list()
cluster_numbers = [2,4,5,7,12]
for n in cluster_numbers:
    np.random.seed(1234)
    h = h_cluster(dat_all, n, showPlot=False)
    h.index = dat_all.index
    count = count_clusters(h.reset_index(), counter='Cluster_hier', grouper1='sample', grouper2 = 'colour')
    count.transpose().plot(kind='bar', stacked=True)
    pl.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title = 'Cluster')
    pl.xlabel('Original image')
    pl.ylabel('Cluster frequency')
    pl.title('Hierarchical clustering')
    pl.show()
    gini_scores.append(np.mean(gini_score(count)))

x = cluster_numbers.copy()
for i in range(13, 33):
    x.append(i)
    np.random.seed(1234)
    h = h_cluster(dat_all, i, showPlot=False)
    h.index = dat_all.index
    count = count_clusters(h.reset_index(), counter='Cluster_hier', grouper1='sample', grouper2 = 'colour')
    gini_scores.append(np.mean(gini_score(count)))

pl.plot(x, gini_scores, label = 'Hierarchical clustering')
pl.xlabel('Number of clusters')
pl.ylabel('Gini score')
pl.legend()
pl.show()
 # %% Gini scores including all possible numbers of clusters, not just the sensible ones
scores2 = list()
x2 = [*range(2,33)]
for i in x2:
    np.random.seed(1234)
    h = h_cluster(dat_all, i, showPlot=False)
    h.index = dat_all.index
    count = count_clusters(h.reset_index(), counter='Cluster_hier', grouper1='sample', grouper2 = 'colour')
    scores2.append(np.mean(gini_score(count)))
pl.plot(x2, scores2, label = 'Hierarchical clustering')
pl.xlabel('Number of clusters')
pl.ylabel('Gini score')
pl.legend()
pl.savefig('/Users/s1101153/Dropbox/Emily/Graphs/hier_gini.png')
pl.show()
