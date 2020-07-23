from slice_tools import *
# from skimage import io

datFile = '/Users/s1101153/Desktop/droplet_stacks/63x/stack_info.csv'

imFolder = '/Users/s1101153/Desktop/droplet_stacks/63x/final_images/stacks/'

# imarr = io.imread(imFolder+'SUM_phip0-5_phir10_2_stack.tif')

# imarr.shape
# io.imshow(imarr[0])
# io.imshow(imarr[1])
# io.imshow(imarr[1,:,:,0])
# io.imshow(imarr[0,:,:,1])

slice_all(datFile=datFile, filePath = imFolder, save=True)
