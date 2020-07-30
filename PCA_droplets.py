from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as pl
from processing_tools import * # also imports all from data_tools and slice_tools
import pandas as pd
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


# %% PCA at 95% variance

pca=PCA(.95)
pca.fit(scaled_vals)
pca.n_components_

transformed_vals=pca.transform(scaled_vals)
print(scaled_vals.shape)
print(transformed_vals.shape)
# we've reduced the number of variables from 27300 to 145

approximation = pca.inverse_transform(transformed_vals)

pl.imshow(scaled_vals[0].reshape(273,100))

pl.imshow(scaled_vals[0].reshape(273,100))


# %% PCA at varying variances

variances = [0.99, 0.97, 0.95, 0.92, 0.9, 0.8, 0.7, 0.5]
n=len(variances)+1
pl.subplot(1,n,1)
pl.imshow(scaled_vals[0].reshape(273, 100))
pl.xlabel('27300 components')
pl.title('Original image')


i=2
for var in variances:
    pca=PCA(var)
    transformed = pca.fit_transform(scaled_vals)
    approximation = pca.inverse_transform(transformed)
    pl.subplot(1,n,i)
    pl.imshow(approximation[0].reshape(273,100))
    pl.xlabel(str(pca.n_components_)+ ' components')
    pl.title(str(var*100) + '% variance')
    i=i+1
pl.show()


# %%

pca=PCA()
pca.fit(scaled_vals)
pca.n_components_
tot=sum(pca.explained_variance_)
tot

var_exp = [i/tot for i in sorted(pca.explained_variance_, reverse=True)]
print(var_exp[0:5])

cum_var_exp = np.cumsum(var_exp)

pl.step(range(1, 385), cum_var_exp, where='mid',label='cumulative explained variance')
pl.title('Cumulative Explained Variance as a Function of the Number of Components')
pl.ylabel('Cumulative Explained variance')
pl.xlabel('Principal components')
pl.axhline(y = .99, color='k', linestyle='--', label = '99% Explained Variance')
pl.axhline(y = .95, color='c', linestyle='--', label = '95% Explained Variance')
pl.axhline(y = .90, color='g', linestyle='--', label = '90% Explained Variance')
pl.axhline(y = .85, color='r', linestyle='--', label = '85% Explained Variance')
pl.legend(loc='best')
pl.show()


# %%

np.random.seed(1234)
kmeans_normal = KMeans(n_clusters = 16).fit(scaled_vals)
labs = pd.DataFrame(kmeans_normal.labels_, index=vals.index, columns=['normal'])


np.random.seed(1234)
kmeans_PCA = KMeans(n_clusters=16).fit(transformed_vals)
labs['PCA'] = kmeans_PCA.labels_
labs
