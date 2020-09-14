from full_analysis_tools import format_rectangles
from slice_tools import read_files
import pandas as pd
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as pl

filePath = '/Users/s1101153/Desktop/droplet_stacks/63x/rect_pickles'
imagePath = '/Users/s1101153/Desktop/droplet_stacks/63x/final_images/ims_to_read/'

dat = read_files(filePath)
r_dat = format_rectangles(dat, scale='standard', theta_av=True)


def optimalK(data, nrefs=3, maxClusters=15):
    """
    FROM https://anaconda.org/milesgranger/gap-statistic/notebook
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):

            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = cluster.KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = cluster.KMeans(k)
        km.fit(data)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k,
                                      'gap': gap},
                                     ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


gap_count = optimalK(r_dat, maxClusters=30)
gap_count[1]

pl.scatter(x=gap_count[1]['clusterCount'], y=gap_count[1]['gap'])
pl.xlabel('Number of clusters')
pl.ylabel('Gap statistic')
pl.show()
