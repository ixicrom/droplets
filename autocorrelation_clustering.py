from slice_tools import *
# import os
import numpy as np
import matplotlib.pyplot as pl
# import scipy.fftpack as fftim
import pandas as pd
# from skimage import io
# import math
from scipy.signal import correlate2d
from sklearn import cluster, preprocessing
from datetime import datetime

# def r_average(data):
#     y, x = np.indices((data.shape))
#     tbin = np.bincount(x.ravel(),data.ravel())
#     nx = np.bincount(x.ravel())
#     rav = tbin/nx
#     rav = rav[:int(len(x)/2)]
#     return rav


def main():
    filePath='/Users/s1101153/Dropbox/Emily/'
    dat=read_files(filePath)
    dat = dat.dropna()

    idx=pd.IndexSlice

# rescale the data______________________________________________
    x = dat.loc[:,idx[:,['val_green','val_red']]]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    dat.loc[:,idx[:,['val_green','val_red']]] = x_scaled
# ____________________________________________________________

    sliceNames = dat.columns
    n_slice = int(len(sliceNames)/4)
    ACF = pd.DataFrame()
    for i in range(n_slice):
        index_g = sliceNames[i*4+2]
        index_r = sliceNames[i*4+3]
        name = index_g[0]
        print(i)
        slice_df = dat.loc[:,name]
        num_r = int(len(slice_df)/100)
        im_g = np.zeros((num_r,100))
        im_r = np.zeros((num_r,100))
        r, t = 0, 0

        for j in range(len(slice_df)):
            im_g[r,t]=slice_df['val_green'][j]
            im_r[r,t]=slice_df['val_red'][j]
            if t==99:
                t=0
                r = r+1
            else:
                t = t+1
        # pl.imshow(im_r)
        ac3_r = correlate2d(im_r,im_r)
        # pl.imshow(ac3_r)
        # pl.imshow(im_g)
        ac3_g = correlate2d(im_g,im_g)
        # pl.imshow(ac3_g)
        # r_average(ac3_g)[:100]

        ACF[index_r]=np.mean(ac3_r[:244, 99:102])

        ACF[index_g]=np.mean(ac3_g[:244, 99:102])


    ACF = ACF.transpose()
    ACF.index = pd.MultiIndex.from_tuples(ACF.index)

    np.random.seed(1234)
    kmeans=cluster.KMeans().fit(ACF)
    labs = pd.Series(kmeans.labels_, index = ACF.index, name='Cluster')

    dat_results = pd.concat([ACF, labs], axis=1)
    dat_results = dat_results.set_index('Cluster', append = True)

    dat_toPlot = dat_results.stack().reset_index()
    dat_toPlot.columns = ['Slice', 'Colour', 'Cluster', 'ACF_index', 'Value']
    dat_toPlot['Sample'] = dat_toPlot['Slice'].str.slice(0,8)

    date = datetime.today().strftime('%Y-%m-%d')
    filePath = '/Users/s1101153/Dropbox/Emily/Results/'

    fileName = date + '_ACF_clusters_1d-av.csv'
    outFile = filePath+fileName
    dat_toPlot.to_csv(outFile, header=True)
    print('File saved: '+outFile)


    fileName = date + '_ACF_clusters_3pt-av.csv'
    out2 = filePath + fileName
    # dat_clusters = dat_toPlot.drop(['ACF_index','Value'], axis='columns').drop_duplicates()
    # dat_clusters = dat_clusters.reset_index(drop=True)
    # dat_clusters.to_csv(out2, header=True)
    print('File saved: '+out2)


main()
# ACF['T6M_29_1_slice5.pkl red'].describe()
# ACF['T3M_7_1_slice8.pkl red'].describe()
# pl.plot(ACF['T3M_7_1_slice8.pkl red'])
# pl.plot(ACF['T6M_29_1_slice5.pkl red'])
# pl.show()
