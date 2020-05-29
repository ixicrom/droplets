from slice_tools import *
# import os
import numpy as np
import matplotlib.pyplot as pl
# import scipy.fftpack as fftim
import pandas as pd
# from skimage import io
# import math
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

    green_images = {}
    red_images = {}
    vars = pd.DataFrame()

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



        green_images[name] = im_g
        red_images[name] = im_r

        mean_r = np.mean(im_r)
        mean_g = np.mean(im_g)


        thresholded_r = np.copy(im_r)
        thresholded_r[thresholded_r > 0.35] = 1
        thresholded_r[thresholded_r <= 0.35] = 0
        naive_a_frac_r = np.sum(thresholded_r)/len(im_r)
        thresholded_g = np.copy(im_g)
        thresholded_g[thresholded_g > 0.35] = 1
        thresholded_g[thresholded_g <= 0.35] = 0
        naive_a_frac_g = np.sum(thresholded_g)/len(im_g)

        vars[index_r] = [mean_r, naive_a_frac_r]
        vars[index_g] = [mean_g, naive_a_frac_g]
    pl.imshow(thresholded_r)
    pl.imshow(im_r)

    np.mean(im_r)

    vars


main()
