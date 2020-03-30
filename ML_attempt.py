from slice_tools import slice_image
# import os
import glob
import pandas as pd
from sklearn import cluster
import numpy as np

def make_files():
    filePath='/Users/s1101153/Dropbox/Emily/'
    datFile=filePath+'z-stack_info.csv'

    f=open(datFile, 'r')

    all_slices=[]
    for line in f.readlines():
        if line.startswith("SUM"):
            vals=line.split(",")
            fileName = filePath+vals[0]
            umSize=float(vals[1])
            pxSize=int(vals[2])
            Cx=float(vals[4])
            Cy=float(vals[3])
            n_slice = 12
            im_slices=slice_image(fileName, umSize, pxSize, Cx, Cy, n_slice, True)

def read_files(folderName):
    search=os.path.join(folderName, "*.pkl")
    file_names=glob.glob(search)#folderName+x for x in os.listdir(folderName)]
    dat=[]
    keys=[]
    for file in file_names:
        entry=pd.read_pickle(file) #actually reading the file
        dat.append(entry)
        # making nice column names_______
        start1=file.find("T")
        end1=file.find("_63xoil")
        part1=file[start1:end1]
        start2=end1+7
        end2=start2+2
        part2=file[start2:end2]
        start3=file.find(".tif")+4
        part3=file[start3:]
        key=part1+part2+part3
        keys.append(str(key))
        # ________________________________
    # print([x for x in keys if keys.count(x) >= 2])
    dat_df=pd.concat(dat, axis=1, keys=keys) #axis=1 for side-by-side. will do multi-layer column names
    return dat_df

def main():
    # make_files()
    dat=read_files("/Users/s1101153/Dropbox/Emily/")
    dat = dat.dropna()
    dat=dat.transpose()

    dat=dat.to_numpy()
    dat.shape

    # type(dat)
    # type(dat[0])
    big_array=np.full((108,4,24400), np.nan)
    for i in range(108):
        j=i*4
        little_array=dat[j:j+4, :]
        big_array[i,]=little_array

    kmeans=cluster.KMeans(n_clusters=8).fit(big_array[:,3,:])
    labs=kmeans.labels_
    len(labs)
    np.savetxt('test.txt',np.asarray(labs))
main()
