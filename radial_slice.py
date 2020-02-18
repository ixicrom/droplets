from skimage import io
import numpy as np
import math
import pandas as pd
# import os

imFile = '/Volumes/PhD/DavidData/Emily/2014_6_2-T2M_6_63xoil_1.lsm'

imArr = io.imread(imFile)
np.shape(imArr[0])

im1 = imArr[0][:,:,0]

L=1024
r=[]
theta=[]
val=[]
for x in range(L):
    for y in range(L):
        x_cent=x-L/2.
        y_cent=y-L/2.
        r.append(np.sqrt(float(x_cent)**2+float(y_cent)**2))
        theta.append(math.atan2(y_cent,x_cent)%(2*math.pi))
        val.append(im1[x,y])

print(r[0:10])
points = pd.DataFrame([r,theta,val]).transpose()
points.columns=['r','theta','val']

points_trimmed = points.loc[points['r']<L/2]
print("Pre-slicing: ")
print(points_trimmed.describe())
slices=[]
for i in range(12):
    theta_min=i/12.*2.*math.pi
    theta_max=(i+1.)/12.*2.*math.pi
    slice_vals = points_trimmed.loc[(points_trimmed['theta']>=theta_min) & (points_trimmed['theta']<theta_max)]
    slice_vals.theta=slice_vals.theta-theta_min
    slices.append(slice_vals)

print("Slice 1: ")
print(slices[0].describe())
print("Slice 12: ")
print(slices[11].describe())
