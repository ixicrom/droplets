from processing_tools import *
from sklearn import cluster
from slice_tools import *

imPath = '/Users/s1101153/Desktop/droplet_stacks/63x/final_images/ims_to_read/'
infoFile = '/Users/s1101153/Desktop/droplet_stacks/63x/stack_info.csv'

if input('Re-calculate image slices? y/n: ') == 'y':
    print('Re-calculating slices...')
    slices = slice_folder(imPath, infoFile, save=True, saveFile='/Users/s1101153/Desktop/droplet_stacks/63x/nice_slices.pkl')

else:
    slices = pd.read_pickle('/Users/s1101153/Desktop/droplet_stacks/63x/nice_slices.pkl')

slices
slices_data = calc_variables(slices, highpass=False)

# #
# #
# img=slices.loc['phip0-5_phir10_2','green',4].values[0]
# img
# io.imshow(img)
# img_good=slices.loc['phip0-5_phir10_2','green',2].values[0]
# io.imshow(img_good)
#
# lab_good=threshold_and_label(img_good, highpass=False)
# np.max(lab_good)
# lab=threshold_and_label(img, highpass=False)
# np.max(lab)
#
# thr=filters.threshold_otsu(img)
# thr
# io.imshow(img>0.5)
#
# thr_good=filters.threshold_otsu(img_good)
# io.imshow(img_good>thr_good)
#
# measure.regionprops(lab_good)
# measure.regionprops(lab)
#

slices_data=slices_data.drop('imArray', axis=1)


norm = False
filePath='/Users/s1101153/Desktop/droplet_stacks/63x/rect_pickles'
dat=read_files(filePath)
idx=pd.IndexSlice

# rescale the data______________________________________________
if(input('Normalise the data? (y/n): ')=='y'):
    norm = True
    dat = norm_data(dat)
# ____________________________________________________________




# remove a dimension by averaging over theta
dat_all = theta_average(dat)
dat_all
dat_all = dat_all.transpose().reset_index()
dat_all['slices']

# # average over groups of N r values to bin data
# N = int(input('How many r values to average over: '))
# dat_r_bins = dat_all.groupby(np.arange(len(dat_all))//N).mean()
# dat_r_bins
# dat_all = dat_r_bins.transpose()
# dat_all
samples=dat_all['slices'].str.slice(4,21).str.rstrip('_stack')
samples
dat_all.insert(0,'sample', samples)
slice_nums=dat_all['slices'].str.rstrip('.pkl').str[-2:].str.lstrip('e').astype(int)
slice_nums
dat_all['slices']=slice_nums
dat_all.rename(columns={'slices':'slice'}, inplace=True)


# do I need to remove this bit? Although if I don't use this code much later then it doesn't matter
x=dat_all['sample']
change_bool = (x=='T6M_30_1') | (x=='T6M_29_1')

cols=dat_all['vars'].str.lstrip('val_')

green_change_bool=(change_bool) & (cols=='green')

red_change_bool=(change_bool) & (cols=='red')



new_cols=cols.copy()



new_cols[green_change_bool] = 'red'
new_cols[red_change_bool] = 'green'

dat_all.insert(1,'colour',new_cols)

# dat_all

dat_all=dat_all.set_index(['sample', 'colour', 'slice'])

# dat_all
# dat_all.sort_index()
# slices_data.sort_index()


full_data = pd.concat([slices_data, dat_all], sort=True, axis=1).drop('vars', axis=1)
full_data

clean_data = full_data.dropna()



np.random.seed(1234)
kmeans_all = cluster.KMeans(n_clusters=8).fit(clean_data)
labels_all =  pd.DataFrame(kmeans_all.labels_, index = clean_data.index, columns=['cluster_all'])
labels_all

toPlot = labels_all.reset_index()
toPlot

cluster_count = count_clusters(toPlot, counter='cluster_all', grouper1='sample', grouper2='colour')
print(gini_score(cluster_count))
print(np.mean(gini_score(cluster_count)))
cluster_count.transpose().plot(kind='bar', stacked=True)
pl.show()


scores = list()
for i in np.arange(1,32):
    np.random.seed(1234)
    kmeans = cluster.KMeans(n_clusters=i).fit(clean_data)
    labels = pd.DataFrame(kmeans.labels_, index=clean_data.index, columns=['cluster'])
    toCount = labels.reset_index()
    clust_count = count_clusters(toCount, counter='cluster', grouper1='sample', grouper2='colour')
    score = np.mean(gini_score(clust_count))
    scores.append(score)


pl.plot(np.arange(1,32),scores)
pl.xlabel('Number of clusters')
pl.ylabel('Gini score')
pl.savefig('/Users/s1101153/Dropbox/Emily/Plots/k-means_scree.svg')
pl.show()


def hier_clusters(num_clust):
    np.random.seed(1234)
    hier = h_cluster(clean_data, num_clust, showPlot=False)
    hier
    hier.index = clean_data.index
    hier
    hier.sort_values(by='Cluster_hier')

    samp_names = hier.index.get_level_values(0)

    phip=samp_names.str[0:7].str.rstrip("_phi")

    phir=samp_names.str[6:15]
    phir=phir.str.lstrip("5_").str.rstrip("_2")

    phir
    hier['phir'] = phir
    hier['phip'] = phip
    hier
    phir_count = count_clusters(hier.reset_index(), counter='Cluster_hier', grouper1 = 'phir', grouper2='colour')
    phip_count = count_clusters(hier.reset_index(), counter='Cluster_hier', grouper1 = 'phip', grouper2='colour')
    phir_count.transpose().plot(kind='bar', stacked=True)
    phip_count.transpose().plot(kind='bar', stacked=True)

def k_means_clusters(num_clust):
    np.random.seed(1234)
    kmeans = cluster.KMeans(n_clusters = num_clust).fit(clean_data)
    labels = pd.DataFrame(kmeans.labels_, index=clean_data.index, columns=['Cluster_kmeans'])
    samp_names = labels.index.get_level_values(0)
    phip=samp_names.str[0:7].str.rstrip("_phi")
    phir=samp_names.str[6:15]
    phir=phir.str.lstrip("5_").str.rstrip("_2")

    # phir=phir.str.lstrip('phir')
    # phip=phip.str.lstrip('phip').str.replace('-','.')
    labels['phir']=phir
    labels['phip']=phip
    phir_count = count_clusters(labels.reset_index(), counter='Cluster_kmeans', grouper1 = 'phir')
    phip_count = count_clusters(labels.reset_index(), counter='Cluster_kmeans', grouper1 = 'phip')
    phir_count.transpose().plot(kind='bar', stacked=True)
    phip_count.transpose().plot(kind='bar', stacked=True)

labels
for i in range(1,10):
    # print('Number of clusters = '+str(i))
    hier_clusters(i)
    # print("_________________________________________________________________")


for i in range(1,10):
    # print('Number of clusters = '+str(i))
    k_means_clusters(i)
    # print("_________________________________________________________________")
