import numpy as np
import matplotlib.pyplot as pl
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis)

# mnist=datasets.fetch_openml('mnist_784')
# X=mnist.data
# y=mnist.target

digits = datasets.load_digits(n_class=10)
X=digits.data
y=digits.target

n_samples, n_features = X.shape
n_neighbors = 30


def plot_labels(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    pl.figure()
    ax = pl.subplot(111)
    for i in range(X.shape[0]):
        pl.text(X[i, 0], X[i, 1], str(y[i]),
                 color=pl.cm.Set1(float(y[i]) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        pl.title(title)
# %% PCA into 2D

X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
X_pca.shape
plot_labels(X_pca, 'PCA')


# %% t-SNE projection

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=100)
X_tsne = tsne.fit_transform(X)
plot_labels(X_tsne, 't-SNE')
