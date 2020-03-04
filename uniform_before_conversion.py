from skimage import io
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as pl

#read the image file into an array
imFile = '/Volumes/PhD/DavidData/Emily/2014_6_2-T2M_6_63xoil_1.lsm'
imArr = io.imread(imFile)
# np.shape(imArr[0])
im1 = imArr[0][:,:,0]

L=1024

x_dat=[]
y_dat=[]
r_dat=[]
theta_dat=[]
val_dat=[]
for r in range(512):
    for i in range(1200):
        theta=i/1200*2*math.pi
        x_cent=r*math.cos(theta)
        y_cent=r*math.sin(theta)
        x=x_cent+L/2
        y=y_cent+L/2
        x_dat.append(round(x))
        y_dat.append(round(y))
        theta_dat.append(theta)
        r_dat.append(r)
        val_dat.append(im1[round(x),round(y)])

pl.scatter(x_dat, y_dat,c=val_dat)


pl.scatter(r_dat, theta_dat, c=val_dat)
        
