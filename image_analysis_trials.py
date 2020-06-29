# %% load packages
from skimage import filters, io, draw, transform, measure, segmentation, color, morphology
from skimage.future import graph
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as pl
import math
import pandas as pd
import matplotlib.patches as mpatches
from sklearn import preprocessing
from scipy import ndimage

# %% read in file
imFile = '/Users/s1101153/Desktop/TIFs/SUM_2014_5_30-T3M_7_63xoil_1.tif'

imArr = io.imread(imFile)
for i in range(3):
    imArr[:,:,i] = imArr[:,:,i]/np.max(imArr[:,:,i])
min_max_scaler = preprocessing.MinMaxScaler()

image = imArr.copy()[:,:,2]

image2 = imArr.copy()[:,:,0]

io.imshow(image)
pl.show()

# %% create mask for segment slice, using coords for centre of droplet
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

# %% use mask to slice image into 12 slices
green_slices = list()
red_slices = list()

n_slices = 12
im = imArr.copy()
for i in range(n_slices):
    masked_im = im.copy()
    masked_im[~mask] = 0#float('NaN')
    masked_im.shape
    green_slice = masked_im[:,:,0]
    red_slice = masked_im[:,:,2]
    # green_slice = green_slice[~np.all(np.isnan(green_slice), axis=1)]
    # green_slice = green_slice[:,~np.all(np.isnan(green_slice), axis=0)]
    # red_slice = red_slice[~np.all(np.isnan(red_slice), axis=1)]
    # red_slice = red_slice[:,~np.all(np.isnan(red_slice), axis=0)]
    pl.imshow(red_slice)
    pl.show()
    # pl.savefig(str(i))
    # pl.clf()
    green_slices.append(green_slice)
    red_slices.append(red_slice)
    im = transform.rotate(im, np.degrees(theta1), center = [C_y, C_x])


# trying different methods of thresholding the image
# %% global thresholding using otsu
threshold_val = filters.threshold_otsu(image)

labeled_foreground = (image>threshold_val)#.astype(int)
io.imshow(labeled_foreground)
pl.show()
io.imshow(image)
pl.show()
area_frac = np.sum(labeled_foreground)
print(area_frac)

image2 = imArr.copy()[:,:,0]
thr_global2 = filters.threshold_otsu(image2)
binary_global2 = image2>thr_global2
io.imshow(binary_global2)
pl.show()
io.imshow(image2)
pl.show()


# %% local thresholding, trying different definitions of 'local'
for i in range(20):
    block_size = i*10 +1
    thr_local = filters.threshold_local(image, block_size, offset=10)
    binary_local = image<thr_local
    pl.imshow(binary_local)
    pl.title(block_size)
    pl.show()

pl.imshow(image2)
for i in range(20):
    block_size = i*10 +1
    thr_local2 = filters.threshold_local(image2, block_size, offset=10)
    binary_local2 = image2<thr_local2
    pl.imshow(binary_local2)
    pl.title(block_size)
    pl.show()


# %% trying image segmentation
labels1 = segmentation.slic(image.astype('double'), compactness=30, n_segments=400)
out1 = color.label2rgb(labels1, image, kind='avg')

g = graph.rag_mean_color(image, labels1, mode='similarity')
labels2 = graph.cut_normalized(labels1, g)
out2 = color.label2rgb(labels2, image, kind='avg')

fig, ax = pl.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

ax[0].imshow(out1)
ax[1].imshow(out2)

for a in ax:
    a.axis('off')

pl.tight_layout()
pl.show()


# %% high-pass filtering (and random little plotting function)
def plot(data, title):
    plot.i += 1
    pl.subplot(3,2,plot.i)
    pl.imshow(data)
    # pl.gray()
    pl.title(title)

plot.i = 0
plot(image, 'Original1')
plot(image2, 'Original2')
g = filters.gaussian(image, sigma=5)
g2 = filters.gaussian(image2, sigma=5)
plot(g, 'Gaussian1')
plot(g2, 'Gaussian2')
hp1 = image-g
hp2 = image2-g2
plot(hp1, 'HighPass1')
plot(hp2, 'HighPass2')
pl.show()

# %% otsu thresholding on high-pass filtered images
thr_hp1 = filters.threshold_otsu(hp1)
bin_hp1 = hp1>thr_hp1
io.imshow(bin_hp1)
pl.show()

thr_hp2 = filters.threshold_otsu(hp2)
bin_hp2 = hp2>thr_hp2
io.imshow(bin_hp2)
pl.show()

# %% calculating circularity of shape from otsu thresholded image using region properties
# will need to think more about this, circularity of the droplet might not make sense in the slices
label_img = measure.label(image)
io.imshow(label_img)
pl.show()

label_otsu_image = measure.label(labeled_foreground)
print(np.unique(label_otsu_image))
io.imshow(label_otsu_image)
pl.show()
io.imshow(labeled_foreground)
pl.show()

label_hp_thr_image = measure.label(bin_hp1)
io.imshow(label_hp_thr_image)
pl.show()


label_props = measure.regionprops_table(label_otsu_image, properties=('filled_area','perimeter','centroid'))

otsu_region_props = pd.DataFrame(label_props)
# print(otsu_region_props)
# biggest_region = otsu_region_props[otsu_region_props['filled_area']==max(otsu_region_props['filled_area'])]
# circularity = 4*math.pi*biggest_region['filled_area']/biggest_region['perimeter']**2
otsu_region_props['circularity'] = 4*math.pi*otsu_region_props['filled_area']/otsu_region_props['perimeter']**2
print(otsu_region_props)

raw_props = measure.regionprops_table(image.astype(int), properties=('filled_area', 'perimeter', 'centroid'))
pd.DataFrame(raw_props)

# %% understanding region labelling better

io.imshow(image)
pl.show()
thresh=filters.threshold_otsu(image)
# plain_thresh_im = image>thresh
# io.imshow(plain_thresh_im)
# pl.show()
bw = morphology.closing(image>thresh, morphology.square(2))
io.imshow(bw)
pl.show()
cleared = segmentation.clear_border(bw)
io.imshow(cleared)
pl.show

label_image = measure.label(cleared)
image_label_overlay = color.label2rgb(label_image, image=image, bg_label=0)
io.imshow(image_label_overlay)
pl.show()
fig, ax = pl.subplots(figsize=(10,6))
ax.imshow(image_label_overlay)

import matplotlib.patches as mpatches
for region in measure.regionprops(label_image):
    if region.area >= 0:
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc-minc, maxr-minr, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
ax.set_axis_off()
pl.tight_layout()
pl.show()




# %% function to apply the above to any image
def threshold_and_label(img, regionsize, highpass = False, drawbox = True):
    if highpass:
        kernel = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])
        img = ndimage.convolve(img, kernel)
    thresh=filters.threshold_otsu(img)
    plain_thresh_im = img>thresh
    bw = morphology.closing(img>thresh, morphology.square(2))

    cleared = segmentation.clear_border(bw)

    label_image = measure.label(cleared)
    image_label_overlay = color.label2rgb(label_image, image=img, bg_label=0)

    fig, ax = pl.subplots(figsize=(10,6))
    ax.imshow(image_label_overlay)

    for region in measure.regionprops(label_image):
        if (region.area >= regionsize and drawbox):
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc-minc, maxr-minr, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    ax.set_axis_off()
    pl.tight_layout()
    pl.show()


threshold_and_label(image, 100)
threshold_and_label(image2, 100)
threshold_and_label(green_slices[0],100)
threshold_and_label(red_slices[0],100)
