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
n_theta=100
n_slices=12
slices=[]
for slice in range(n_slices):
    # x_dat=[]
    # y_dat=[]
    r_dat=[]
    theta_dat=[]
    val_dat=[]
    for r in range(int(L/2)):
        for i in range(n_theta):
            theta=2*math.pi/n_slices * i/n_theta
            theta_calc=theta + slice*2*math.pi/n_slices
            x_cent=r*math.cos(theta_calc)
            y_cent=r*math.sin(theta_calc)
            x=x_cent+L/2
            y=y_cent+L/2
            # x_dat.append(round(x))
            # y_dat.append(round(y))
            theta_dat.append(theta)
            r_dat.append(r)
            val_dat.append(im1[round(x),round(y)])
    points = pd.DataFrame([r_dat,theta_dat,val_dat]).transpose()
    points.columns=['r','theta','val']
    slices.append(points)

print("Slice 1: ")
print(slices[0].describe())
print("Slice 12: ")
print(slices[11].describe())
# pl.scatter(x_dat, y_dat,c=val_dat)
pl.scatter(slices[0]['r'], slices[0]['theta'], c=slices[0]['val'])

for slice in range(12):
    plot_file="/Volumes/PhD/DavidData/Plots/T2M_6_1_newSlice"+str(slice+1)+".png"
    pl.scatter(slices[slice]['r'], slices[slice]['theta'], c=slices[slice]['val'])
    pl.xlabel("r (pixels)")
    pl.ylabel("theta (rad)")
    pl.title("Slice "+str(slice+1))
    pl.savefig(plot_file)
