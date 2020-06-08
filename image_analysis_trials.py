from skimage import filters, io, draw, transform, measure
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as pl
import math

imFile = '/Users/s1101153/Desktop/TIFs/SUM_2014_5_30-T3M_7_63xoil_1.tif'

imArr = io.imread(imFile)
imArr.shape

io.imshow(imArr[:,:,2])

L = 2048
um = 72.58
pixel_um=L/um
C_x_um = 30.195
C_y_um = 34.342


C_y=C_x_um*pixel_um
C_x=C_y_um*pixel_um
r = round(min([C_x, C_y, L-C_x, L-C_y]))

theta0, theta1 = 0, math.pi/6

x1, y1 = C_x - 1.5 * r * np.sin(theta0), C_y + 1.5 * r * np.cos(theta0)
x2, y2 = C_x - 1.5 * r * np.sin(theta1), C_y + 1.5 * r * np.cos(theta1)


mask_circle = np.zeros(imArr.shape[:2], dtype = bool)
mask_poly = np.zeros(imArr.shape[:2], dtype = bool)

xx, yy = draw.circle(C_x, C_y, r, shape = mask_circle.shape)
mask_circle[xx, yy] = 1

xx, yy = draw.polygon([C_x, x1, x2, C_x], [C_y, y1, y2, C_y], shape = mask_poly.shape)
mask_poly[xx,yy] = 1

mask = mask_circle & mask_poly
#
# pl.imshow(mask)
#
#
#
# im_masked = imArr.copy()
# im_masked[~mask] = 0
# im_masked.shape
# pl.imshow(im_masked)
#
# slice2 = transform.rotate(imArr, np.degrees(theta1), center = [C_x, C_y])
# io.imshow(slice2)
#
# slice2.shape
# slice2_masked = slice2.copy()
# slice2_masked[~mask] = 0
#
#
#
# pl.imshow(slice2_masked)

green_slices = list()
red_slices = list()

n_slices = 12
im = imArr.copy()
for i in range(n_slices):
    masked_im = im.copy()
    masked_im[~mask] = float('NaN')
    masked_im.shape
    green_slice = masked_im[:,:,0]
    red_slice = masked_im[:,:,2]
    green_slice = green_slice[~np.all(np.isnan(green_slice), axis=1)]
    green_slice = green_slice[:,~np.all(np.isnan(green_slice), axis=0)]
    red_slice = red_slice[~np.all(np.isnan(red_slice), axis=1)]
    red_slice = red_slice[:,~np.all(np.isnan(red_slice), axis=0)]
    pl.imshow(im)
    pl.savefig(str(i))
    pl.clf()
    green_slices.append(green_slice)
    red_slices.append(red_slice)
    im = transform.rotate(im, np.degrees(theta1), center = [C_y, C_x])



# threshold image
image = imArr.copy()[:,:,0]
threshold_val = filters.threshold_otsu(image)

labeled_foreground = (image>threshold_val).astype(int)
io.imshow(labeled_foreground)
io.imshow(image)
