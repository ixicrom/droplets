from skimage import io
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as pl

#read the image file into an array
imFile = '/Volumes/PhD/DavidData/Emily/2014_6_2-T2M_6_63xoil_1.lsm'
imArr = io.imread(imFile, plugin = 'tifffile')
imArr.shape
im1 = imArr[0][0][:,:,0]
pl.imshow(im1)
#initialise arrays and set image size L in pixels
L=1024
r=[]
theta=[]
val=[]
#fill arrays with values and positions converted to r and theta
for x in range(L):
    for y in range(L):
        x_cent=x-L/2.
        y_cent=y-L/2.
        r.append(np.sqrt(float(x_cent)**2+float(y_cent)**2))
        theta.append(math.atan2(y_cent,x_cent)%(2*math.pi))
        val.append(im1[x,y])

# print(r[0:10])

#turn arrays into a single dataframe
points = pd.DataFrame([r,theta,val]).transpose()
points.columns=['r','theta','val']

#cut off corners so we are left with a circular image and all slices can be the same shape
points_trimmed = points.loc[points['r']<L/2]
print("Pre-slicing: ")
print(points_trimmed.describe())

#split image into 12 slices
slices=[]
for i in range(12):
    plot_file="/Volumes/PhD/DavidData/Plots/T2M_6_1_slice"+str(i)+".png"
    plot_file_2d="/Volumes/PhD/DavidData/Plots/T2M_6_1_slice"+str(i)+"_2d.png"
    array_file="/Volumes/PhD/DavidData/Slices/T2M_6_1_slice"+str(i)+".npy"
    theta_min=i/12.*2.*math.pi
    theta_max=(i+1.)/12.*2.*math.pi
    slice_vals = points_trimmed.loc[(points_trimmed['theta']>=theta_min) & (points_trimmed['theta']<theta_max)]
    slice_vals.theta=slice_vals.theta-theta_min
    np.save(array_file, slice_vals)
    slices.append(slice_vals)
    pl.scatter(slice_vals["r"],slice_vals["val"])
    pl.xlabel("r (pixels)")
    pl.ylabel("Pixel value")
    pl.savefig(plot_file)
    pl.clf()

    pl.scatter(slice_vals["r"], slice_vals["theta"], c=slice_vals["val"])
    pl.xlabel("r (pixels)")
    pl.ylabel("theta (radians)")
    pl.savefig(plot_file_2d)
    pl.clf()

#see what the slice arrays look like, check first and last
print("Slice 1: ")
print(slices[0].describe())
print("Slice 12: ")
print(slices[11].describe())

#save each slice as a numpy array
# slices[0]["r"]
