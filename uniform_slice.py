import numpy as np
import math
import matplotlib.pyplot as pl


slice=np.array(np.load("/Volumes/PhD/DavidData/Slices/T2M_6_1_slice1.npy"))
slice[:,2]


grid_theta=[]
grid_r=[]
grid_vals=[]
for i in range(100):
    for j in range(100):
        theta=j/100.*math.pi/6.
        r=i/100*512
        distances=np.sqrt((slice[:,0]-r)**2+(slice[:,1]-theta)**2)
        val_pos = np.argmin(distances)
        val=slice[val_pos,2]
        grid_theta.append(theta)
        grid_r.append(r)
        grid_vals.append(val)

pl.scatter(grid_r, grid_theta,c=grid_vals)
