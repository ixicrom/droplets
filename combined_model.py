from processing_tools import *
from sklearn import cluster

imPath = '/Users/s1101153/Desktop/TIFs/'
infoFile = '/Users/s1101153/Dropbox/Emily/z-stack_info.csv'

if input('Re-calculate image slices? y/n: ') == 'y':
    print('Re-calculating slices...')
    slices = slice_folder(imPath, infoFile, save=False)

else:
    slices = pd.read_pickle('/Users/s1101153/Dropbox/Emily/nice_slices.pkl')

slices_data = calc_variables(slices, highpass=False)

slices_data=slices_data.drop('imArray', axis=1)


norm = False
filePath='/Users/s1101153/Dropbox/Emily/rect_pickles'
dat=read_files(filePath)
idx=pd.IndexSlice

# rescale the data______________________________________________
if(input('Normalise the data? (y/n): ')=='y'):
    norm = True
    dat = norm_data(dat)
# ____________________________________________________________




# remove a dimension by averaging over theta
dat_all = theta_average(dat)
# dat_all
dat_all = dat_all.transpose().reset_index()


# # average over groups of N r values to bin data
N = int(input('How many r values to average over: '))
dat_r_bins = dat_all.groupby(np.arange(len(dat_all))//N).mean()
dat_all = dat_r_bins.transpose()

dat_all.insert(0,'sample', dat_all['slices'].str.slice(0,8).str.rstrip('_'))
dat_all['slices']=dat_all['slices'].str.rstrip('.pkl').str[-2:].str.lstrip('e').astype(int)
dat_all.rename(columns={'slices':'slice'}, inplace=True)

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
