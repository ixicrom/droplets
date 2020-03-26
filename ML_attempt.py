from slice_tools import slice_image
import os
import pandas as pd
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
    file_names=[folderName+x for x in os.listdir(folderName)]
    dat=[]
    keys=[]
    for file in file_names:
        entry=pd.read_pickle(file)
        dat.append(entry)
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
    print([x for x in keys if keys.count(x) >= 2])
    dat_df=pd.concat(dat, axis=1, keys=keys) #axis=1 for side-by-side. will do multi-layer column names
    return dat_df

def main():
    # make_files()

    dat=read_files("/Users/s1101153/Dropbox/Emily/slices/")
    print(dat.head())

main()
dat=pd.read_pickle('/Users/s1101153/Dropbox/Emily/slices/SUM_2014_5_30-T3M_7_63xoil_1.tif_slice0')
print(dat)
folderName="/Users/s1101153/Dropbox/Emily/slices/"
test_file='SUM_2014_5_30-T3M_7_63xoil_1.tif_slice0'
start1=test_file.find("T")
end1=test_file.find("_63xoil")
part1=test_file[start1:end1]
start2=end1+7
end2=start2+2
part2=test_file[start2:end2]
start3=test_file.find(".tif")+4
part3=test_file[start3:]
test_file=part1+part2+part3
test_file
