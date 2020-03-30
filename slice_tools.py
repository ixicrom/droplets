from skimage import io
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as pl


def slice_image(imFile, um, L, C_x_um, C_y_um, n_slices, saveFile=False):
    imArr = io.imread(imFile)

    imGreen=imArr[:,:,0]
    imRed=imArr[:,:,2]

    n_theta=100 #number of theta points desired per slice
    pixel_um=L/um #conversion from microns to pixels

    #convert centre coordinates to pixels
    C_x=C_x_um*pixel_um
    C_y=C_y_um*pixel_um

    #convert centre coordinates because ImageJ counts from different corner to what the code uses
    # C_x = C_x
    # C_y = C_y

    slices=[] #list to add results to
    for slice in range(n_slices):
        #create empty arrays for each output value for this slice
        r_dat=[]
        theta_dat=[]
        val_dat_g=[]
        val_dat_r=[]
        for r in range(int(L-max(C_x, C_y))): #loop through r values, max is shortest distance to edge of image
            for i in range(n_theta): #loop through how many theta values we want
                theta=2*math.pi/n_slices * i/n_theta #theta for output, values from 0 to 2pi/n_slices
                theta_calc=theta + slice*2*math.pi/n_slices #theta for calculating points, values from 0 to 2pi
                x_cent=r*math.cos(theta_calc) #point to plot, x relative to centre of image
                y_cent=r*math.sin(theta_calc) #point to plot, y relative to centre of image
                #convert points to plot so they're relative to the corner of the image
                x=x_cent+C_x
                y=y_cent+C_y
                #append values to arrays
                theta_dat.append(theta)
                r_dat.append(r)
                val_dat_g.append(imGreen[round(x),round(y)])
                val_dat_r.append(imRed[round(x),round(y)])

        #collect different values into a dataframe, and append it to the slices array
        points = pd.DataFrame([r_dat,theta_dat,val_dat_g, val_dat_r]).transpose()
        points.columns=['r','theta','val_green','val_red']
        slices.append(points)
        if saveFile:
            saveName=imFile+"_slice"+str(slice)+".pkl"
            points.to_pickle(saveName)

    return slices

def slices_plot(n_slices, plot_name, slices):
    for slice in range(n_slices):
        red_file="/Users/s1101153/Desktop/Emily/Plots/23Mar/"+plot_name+"_red"+str(slice+1)+".png"
        green_file="/Users/s1101153/Desktop/Emily/Plots/23Mar/"+plot_name+"_green"+str(slice+1)+".png"

        pl.scatter(slices[slice]['r'], slices[slice]['theta'], c=slices[slice]['val_red'], cmap="Reds")
        pl.xlabel("r (pixels)")
        pl.ylabel("theta (rad)")
        pl.title("Slice "+str(slice+1))
        pl.savefig(red_file)
        pl.close()

        pl.scatter(slices[slice]['r'], slices[slice]['theta'], c=slices[slice]['val_green'], cmap="Greens")
        pl.xlabel("r (pixels)")
        pl.ylabel("theta (rad)")
        pl.title("Slice "+str(slice+1))
        pl.savefig(green_file)
        pl.close()
