from slice_tools import *
from skimage import io
import matplotlib.pyplot as pl

datFile = '/Users/s1101153/Desktop/droplet_stacks/63x/stack_info.csv'

imFolder = '/Users/s1101153/Desktop/droplet_stacks/63x/final_images/stacks/'
infoFile = '/Users/s1101153/Desktop/droplet_stacks/63x/stack_info.csv'

newImFolder = '/Users/s1101153/Desktop/droplet_stacks/63x/final_images/ims_to_read/'

f=open(infoFile,'r')

for line in f.readlines():
    if line.startswith("SUM"):
        vals=line.split(",")
        fileName=imFolder+vals[0]
        imarr = io.imread(fileName)
        new_im = np.full([1025,1025,2],np.NaN)
        new_im[:,:,0]=imarr[0,:,:,1]
        new_im[:,:,1]=imarr[1,:,:,0]

        # io.imshow(new_im[:,:,0])
        # pl.show()
        # io.imshow(new_im[:,:,1])
        # pl.show()
        io.imsave(newImFolder+vals[0],new_im)


im=io.imread(newImFolder+vals[0])
im.shape
