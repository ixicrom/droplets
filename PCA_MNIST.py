from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as pl
import numpy as np

# import mnist data
mnist = fetch_openml('mnist_784')

# split data into test and training sets
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1./7., random_state=0)

# scale data, using the training set to fit the scaler
scaler = StandardScaler()
scaler.fit(train_img)
train_img_scaled = scaler.fit_transform(train_img)
test_img_scaled = scaler.transform(test_img)
test_img_scaled[1]

# initialise the PCA object with a 95% variance cut-off (i.e. keep the n principal components that explain 95% of the variance in the data)
pca=PCA(.95)
pca.fit(train_img_scaled)
pca.n_components_

train_img_transform = pca.transform(train_img_scaled)
test_img_transform = pca.transform(test_img_scaled)

approximation = pca.inverse_transform(train_img_transform)

pl.imshow(train_img_scaled[0].reshape(28,28), cmap=pl.cm.gray)
pl.show()


pl.imshow(approximation[0].reshape(28,28), cmap=pl.cm.gray)
pl.show()
