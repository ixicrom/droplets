from slice_tools import *
import os
import numpy as np
import matplotlib.pyplot as pl
import scipy.fftpack as fftim
# from scipy.misc.pilutil import Image
import pandas as pd
from skimage import io
import math
from scipy.signal import correlate2d
from sklearn import cluster, preprocessing
from datetime import datetime


# def r_average(data):
#     y, x = np.indices((data.shape))
#     tbin = np.bincount(x.ravel(),data.ravel())
#     nx = np.bincount(x.ravel())
#     rav = tbin/nx
#     return rav

# def radial_profile(data,centre):
#     y, x = np.indices((data.shape))
#     r = np.sqrt((x-centre[0])**2+(y-centre[1])**2)
#     r = r.astype(np.int)
#     tbin = np.bincount(r.ravel(),data.ravel())
#     nr = np.bincount(r.ravel())
#     radialprofile = tbin/nr
#     return radialprofile

# def correlate(x,y):
#     fr = fftim.fft2(x)
#     fr2 = fftim.fft2(np.flipud(np.fliplr(y)))
#     m,n = fr.shape
#     cc = np.real(fftim.ifft2(fr*fr2))
#     cc = np.roll(cc, int(-m/2+1),axis=0)
#     cc = np.roll(cc, int(-n/2+1),axis=1)
#     return cc

def main():
    filePath='/Users/s1101153/Dropbox/Emily/'
    dat=read_files(filePath)
    dat = dat.dropna()

    dat


    idx=pd.IndexSlice

# rescale the data______________________________________________
    x = dat.loc[:,idx[:,['val_green','val_red']]]
    x.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    pd.DataFrame(x_scaled).describe()
    dat.loc[:,idx[:,['val_green','val_red']]] = x_scaled
    dat
# ____________________________________________________________

    # r_vals = dat.loc[:,idx[:,'r']]
    #
    # theta_vals = dat.loc[:,idx[:,'theta']]
    # theta_vals
    #
    # imFile = filePath+'SUM_2014_5_30-T3M_7_63xoil_1.tif'
    # slice = slice_image(imFile, 72.58, 2048, 30.195, 34.342, 12, saveFile=False)

    sliceNames = dat.columns
    sliceNames
    n_slice = int(len(sliceNames)/4)
    n_slice
    ACF = pd.DataFrame()
    # i=0
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

        slice_df['val_green'][0]
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
        # pl.imshow(im_g)
        # autoCorr_r = r_average(ac3_r)
        ac3_r.shape
        ac3_g = correlate2d(im_g,im_g)
        # autoCorr_g = r_average(ac3_r)

        # print(col_r)
        ACF[index_r]=ac3_r.ravel()

        # print(col_g)
        ACF[index_g]=ac3_g.ravel()
        if i==1:
            print(ACF)

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

    # fileName = date + 'ACF-clusters.csv'
    # outFile = filePath+fileName
    # dat_toPlot.to_csv(outFile, header=True)
    # print('File saved: '+outFile)


    fileShort = date + '_ACF_clusters_only.csv'
    out2 = filePath + fileShort
    dat_clusters = dat_toPlot.drop(['ACF_index','Value'], axis='columns').drop_duplicates()
    dat_clusters = dat_clusters.reset_index(drop=True)
    dat_clusters.to_csv(out2, header=True)
    print('File saved: '+out2)


main()
ACF['T6M_29_1_slice5.pkl red'].describe()
ACF['T3M_7_1_slice8.pkl red'].describe()
pl.plot(ACF['T3M_7_1_slice8.pkl red'])
pl.plot(ACF['T6M_29_1_slice5.pkl red'])
pl.show()
