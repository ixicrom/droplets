from slice_tools import *
from sklearn import cluster, preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import math
from datetime import datetime


def main():
    # read in files and drop NAs
    filePath='/Users/s1101153/Dropbox/Emily/'
    dat=read_files(filePath)
    dat = dat.dropna()
    idx=pd.IndexSlice

# rescale the data______________________________________________
    x = dat.loc[:,idx[:,['val_green','val_red']]]
    x.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    # pd.DataFrame(x_scaled).describe()
    dat.loc[:,idx[:,['val_green','val_red']]] = x_scaled
# ____________________________________________________________

    im1 = dat[['T2M_6_1_slice1.pkl']]
    im2 = dat[['T2M_6_1_slice2.pkl']]
    im3 = dat[['T2M_6_1_slice3.pkl']]

    x=im3.loc[:,idx[:,'r']]
    y=im3.loc[:,idx[:,'theta']]
    valg=im3.loc[:,idx[:,'val_green']]
    valr=im3.loc[:,idx[:,'val_red']]

    plt.scatter(x,y,c=valg, cmap='Greens')
    plt.xlabel("r (pixels)")
    plt.ylabel("theta (rad)")
    plt.title("Slice 3")
    plt.savefig("T2M_6_1_slice3_norm_g.png")
    plt.close()

    plt.scatter(x,y,c=valr, cmap='Reds')
    plt.xlabel("r (pixels)")
    plt.ylabel("theta (rad)")
    plt.title("Slice 3")
    plt.savefig("T2M_6_1_slice3_norm_r.png")
    plt.close()

    dat
    len(dat)

    N = 10
    dat.index

    im1 = dat_r_bins[['T2M_6_1_slice1.pkl']]
    im2 = dat_r_bins[['T2M_6_1_slice2.pkl']]
    im3 = dat_r_bins[['T2M_6_1_slice3.pkl']]

    im1
    x=im3.loc[:,idx[:,'r']]
    y=im3.loc[:,idx[:,'theta']]
    valg=im3.loc[:,idx[:,'val_green']]
    valr=im3.loc[:,idx[:,'val_red']]

    plt.scatter(x,y,c=valg, cmap='Greens')
    plt.xlabel("r (pixels)")
    plt.ylabel("theta (rad)")
    plt.title("Slice 3")
    plt.savefig("T2M_6_1_slice3_av10_g.png")
    plt.close()

    plt.scatter(x,y,c=valr, cmap='Reds')
    plt.xlabel("r (pixels)")
    plt.ylabel("theta (rad)")
    plt.title("Slice 3")
    plt.savefig("T2M_6_1_slice3_av10_r.png")
    plt.close()

main()
