import imutils
import cv2
import numpy as np
from slice_tools import *
import math


# slice up images, don't want to have to do this every time. Maybe should replace with reading the slices in from files.
filePath='/Users/s1101153/Dropbox/Emily/'
datFile=filePath+'z-stack_info.csv'
images = slice_all(datFile, filePath, save=False)

# just take one of the slices to test on
image=images[0][0]
image
# get list of unique r and theta values
theta=image['theta']
r=image['r']
theta=theta.unique()
r=r.unique()



# make an empty array to be the new image
im_arr = np.empty([len(theta), len(r)], dtype='uint8')

# cycle through r and theta values
for r_val in r:
    for t_val in theta:
        # turn values into integer indices
        r_index = int(r_val)
        t_index = int(t_val/math.pi*6*100)

        # look up pixel value for that location
        val_row=image.loc[(image['r']==r_val) & (image['theta']==t_val)]
        val_green=val_row['val_green']

        # write the pixel value into the image array
        im_arr[t_index,r_index]=val_green

# im_arr
# im_arr.shape

# threshold the image
im_edited = cv2.threshold(im_arr, 10, 255, cv2.THRESH_BINARY_INV)[1]
# cv2.imshow("Image",im_edited)
# find contours in the thresholded image
im_contours = cv2.findContours(im_edited.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
im_contours = imutils.grab_contours(im_contours)
len(im_contours[0])
im_contours
