from slice_tools import *

def main():
    filePath='/Users/s1101153/Dropbox/Emily/'
    datFile=filePath+'z-stack_info.csv'

    f=open(datFile, 'r')

    for line in f.readlines():
        if line.startswith("SUM"):
            vals=line.split(",")
            fileName = filePath+vals[0]
            umSize=float(vals[1])
            pxSize=int(vals[2])
            Cx=float(vals[4])
            Cy=float(vals[3])
            n_slice = 12
            im_slices=slice_image(fileName, umSize, pxSize, Cx, Cy, n_slice)
            plot_name = str(vals[0][4:len(vals[0])-4])
            # slices_plot(n_slice,plot_name, im_slices)
    f.close()
main()
