# %% imports
from skimage import filters, io, draw, transform, measure, segmentation, color, morphology
from skimage.future import graph
import numpy as np
from sklearn import preprocessing, cluster
import matplotlib.pyplot as pl
import math
import pandas as pd
import matplotlib.patches as mpatches
from scipy import ndimage, spatial
from os import path

from slice_tools import *
from data_tools import *


# %% hard inputs for testing
# file='/Users/s1101153/Desktop/TIFs/SUM_2014_5_30-T3M_7_63xoil_1.tif'
# L = 2048
# um = 72.58
# pixel_um=L/um
# C_x_um = 30.195
# C_y_um = 34.342
# g_index=2
# r_index=0

# %%
def read_slice(file, L, um, C_x_um, C_y_um, g_index, r_index, r=None):
    im = io.imread(file)

    file_title = path.splitext(path.split(file)[1])[0]
    start1=file_title.find("T")
    end1=file_title.find("_63xoil")
    part1=file_title[start1:end1]
    start2=end1+7
    end2=start2+2
    part2=file_title[start2:end2]
    samplename=part1+part2

    pixel_um=L/um
    C_y = C_y_um*pixel_um
    C_x = C_x_um*pixel_um

    if r == None:
        r = round(min([C_x, C_y, L-C_x, L-C_y]))

    theta0, theta1 = 0, math.pi/6

    x1, y1 = C_x - 1.5 * r * np.sin(theta0), C_y + 1.5 * r * np.cos(theta0)
    x2, y2 = C_x - 1.5 * r * np.sin(theta1), C_y + 1.5 * r * np.cos(theta1)


    mask_circle = np.zeros(im.shape[:2], dtype = bool)
    mask_poly = np.zeros(im.shape[:2], dtype = bool)

    xx, yy = draw.circle(C_x, C_y, r, shape = mask_circle.shape)
    mask_circle[xx, yy] = 1

    xx, yy = draw.polygon([C_x, x1, x2, C_x], [C_y, y1, y2, C_y], shape = mask_poly.shape)
    mask_poly[xx,yy] = 1

    mask = mask_circle & mask_poly
    theta0, theta1 = 0, math.pi/6
    cols=[None]*24
    cols[0:11]=['green']*12
    cols[12:]=['red']*12
    arrays=[[samplename]*24,cols,[0,1,2,3,4,5,6,7,8,9,10,11]*2]
    ind=pd.MultiIndex.from_arrays(arrays, names=('sample', 'colour', 'slice'))
    dat=pd.DataFrame(index=ind, columns=['imArray'])
    idx=pd.IndexSlice

    n_slices = 12
    i=0
    for i in range(n_slices):
        masked_im = im.copy()
        masked_im[~mask] = 0#float('NaN')

        masked_im.shape
        green_slice = masked_im[:,:,g_index]
        green_slice = green_slice/np.max(green_slice)
        red_slice = masked_im[:,:,r_index]
        red_slice = red_slice/np.max(red_slice)
        dat.loc[idx[samplename,'green',i],'imArray']=green_slice
        dat.loc[idx[samplename,'red',i],'imArray']=red_slice
        im = transform.rotate(im, np.degrees(theta1), center = [C_y, C_x])

        # df = pd.DataFrame([green_slices, red_slices])
    return dat
# dat

def slice_folder(filePath, infoFile, save=False, saveFile=None):
    f = open(infoFile, 'r')
    first_time=True
    for line in f.readlines():
        if line.startswith('SUM'):
            vals=line.split(',')
            fileName = filePath+vals[0]
            umSize=float(vals[1])
            L=int(vals[2])
            Cx=float(vals[4])
            Cy=float(vals[3])
            g_index=int(vals[5]) #call the fuller channel the green one
            r_index=int(vals[6]) #call the sparser channel the red one
            n_slice = 12
            if(first_time):
                slices = read_slice(fileName, L, umSize, Cx, Cy, g_index, r_index)
                first_time=False
            else:
                newslice = read_slice(fileName, L, umSize, Cx, Cy, g_index, r_index)
                slices = pd.concat([slices, newslice], axis=0)
    f.close()
    if save:
        slices.to_pickle(saveFile)
        print('New slices saved: ' + saveFile)
    else:
        print('New slices not saved')
    return slices

def threshold_and_label(img, highpass = False):
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

    return label_image

    # image_label_overlay = color.label2rgb(label_image, image=img, bg_label=0)
    #
    # fig, ax = pl.subplots(figsize=(10,6))
    # ax.imshow(image_label_overlay)
    #
    # for region in measure.regionprops(label_image):
    #     if (region.area >= regionsize and drawbox):
    #         minr, minc, maxr, maxc = region.bbox
    #         rect = mpatches.Rectangle((minc, minr), maxc-minc, maxr-minr, fill=False, edgecolor='red', linewidth=2)
    #         ax.add_patch(rect)
    # ax.set_axis_off()
    # pl.tight_layout()
    # pl.show()


def calc_variables(slices, highpass):
    av_connected_area = list()
    av_separation = list()
    area_sum = list()
    av_circularity = list()
    for s in slices['imArray']:
        lab_im = threshold_and_label(s, highpass = highpass)
        areas=list()
        centres=list()
        circs=list()
        for region in measure.regionprops(lab_im):
            areas.append(region.area)
            centres.append(region.centroid)
            if region.perimeter ==0:
                circ = 0
            else:
                circ = 4*math.pi*region.area/region.perimeter**2
            circs.append(circ)

        av_connected_area.append(np.mean(areas))

        distances = spatial.distance.pdist(centres)
        av_separation.append(np.mean(distances))

        area_sum.append(np.sum(areas))

        av_circularity.append(np.mean(circs))


    #     if len(areas)>0:
    #         print(str(i) + ': ' + str(np.mean(areas)))
    #     else:
    #         print(str(i) + ': 0')
    #         print(str(i) + ': ' + str(np.mean(areas)))
    #     i=i+1

    slices['av_connected_area'] = av_connected_area
    slices['av_separation'] = av_separation
    slices['area_sum'] = area_sum
    slices['av_circularity'] = av_circularity
    return slices
