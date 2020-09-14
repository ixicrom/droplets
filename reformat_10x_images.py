from skimage import io
import matplotlib.pyplot as pl
import numpy as np

imFolder = '/Users/s1101153/Desktop/droplet_stacks/10x/channel_stacks/'
infoFile = '/Users/s1101153/Desktop/droplet_stacks/10x/stack_info_10.csv'

newImFolder = '/Users/s1101153/Desktop/droplet_stacks/10x/channel_stacks/for_code/'

f = open(infoFile, 'r')

for line in f.readlines():
    if line.startswith("SUM"):
        vals = line.split(",")
        fileName = imFolder+vals[0]
        num_pixels = int(vals[2])
        imarr = io.imread(fileName)
        new_im = np.full([num_pixels, num_pixels, 2], np.NaN)
        new_im[:, :, 0] = imarr[0, :, :, 1]
        new_im[:, :, 1] = imarr[1, :, :, 0]
        pl.imshow(new_im[:, :, 0])
        pl.show()
        pl.imshow(new_im[:, :, 1])
        pl.show()
        io.imsave(newImFolder+vals[0], new_im)

f.close()

im = io.imread(newImFolder+vals[0])
im.shape
