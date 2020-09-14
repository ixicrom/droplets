from processing_tools import calc_variables
import pandas as pd
from skimage import filters, morphology, measure, io
import math
import matplotlib.pyplot as pl
import numpy as np
from datetime import datetime

imagePath = '/Users/s1101153/Desktop/droplet_stacks/10x/channel_stacks/for_code/'
infoFile = '/Users/s1101153/Desktop/droplet_stacks/10x/stack_info_10.csv'

green_images = []
red_images = []
im_names = []
f = open(infoFile, 'r')
first_time = True
for line in f.readlines():
    if line.startswith('SUM'):
        vals = line.split(',')
        im_file = imagePath+vals[0]
        im = io.imread(im_file)
        green_images.append(im[:, :, 0])
        red_images.append(im[:, :, 1])
        im_name = vals[0][4:-10]
        im_names.append(im_name)

green_df = pd.DataFrame([im_names, green_images]).transpose()
green_df.columns = ['sample', 'imArray']

red_df = pd.DataFrame([im_names, red_images]).transpose()
red_df.columns = ['sample', 'imArray']

# green_vars = calc_variables(green_df, highpass=False)
# red_vars = calc_variables(red_df, highpass=False)


def calc_circularity(index, col):
    thr_ok = False
    if col == 'r':
        im = red_df['imArray'][index]
    elif col == 'g':
        im = green_df['imArray'][index]
    elif col == 'sum':
        im = red_df['imArray'][index] + green_df['imArray'][index]
    pl.imshow(im)
    pl.title("Original image")
    pl.show()

    thr = filters.threshold_otsu(im)
    print('Otsu threshold: '+str(thr))
    pl.imshow(im < thr)
    pl.title("Otsu thresholded")
    pl.show(block=False)

    while not thr_ok:
        thr_check = input("Is the thresholding ok? yes: y, no: type number to try: ")
        if thr_check == 'y':
            thr_ok = True
            pl.close()
        else:
            pl.close()
            thr = int(thr_check)
            pl.imshow(im < thr)
            pl.title("Threshold = " + str(thr))
            pl.show(block=False)

    thr_im = im < thr

    bw = morphology.closing(thr_im, morphology.square(2))

    labelled_im = measure.label(bw)
    circs = list()
    for region in measure.regionprops(labelled_im):
        if region.area > 1000 and region.area < 50000:
            circ = 4*math.pi*region.area/region.perimeter**2
            circs.append(circ)
    print(str(len(circs))+' droplets found')

    return thr, np.mean(circs)


circularity_green = []
thr_green = []
for i in range(0, 9):
    th, circ = calc_circularity(i, 'g')
    circularity_green.append(circ)
    thr_green.append(th)

circularity_red = []
thr_red = []
for i in range(0, 9):
    th, circ = calc_circularity(i, 'r')
    circularity_red.append(circ)
    thr_red.append(th)

circularity_sum = []
thr_sum = []
for i in range(9):
    sum_df = green_df['imArray'][i]+red_df['imArray'][i]
    th, circ = calc_circularity(i, 'sum')
    circularity_sum.append(circ)
    thr_sum.append(th)

green_df['Circularity_g'] = circularity_green
red_df['Circularity_r'] = circularity_red

green_df = green_df.set_index('sample')
red_df = red_df.set_index('sample')

circ_df = green_df.drop('imArray', axis=1).join(red_df.drop('imArray', axis=1))
circ_df['Circularity_sum'] = circularity_sum

samps = circ_df.index
phip = samps.str[0:7].str.strip('_phi')
phir = samps.str[10:15].str.strip('phir-')

circ_df.insert(0, 'phir', phir.astype(int))
circ_df.insert(0, 'phip', phip.str.replace('-', '.').astype(float))

circ_df

circ_05 = circ_df[circ_df['phip'] == 0.5]
circ_1 = circ_df[circ_df['phip'] == 1.0]

pl.plot(circ_05['phir'], circ_05['Circularity_g'],
        marker='o', color='green',
        label='phip=0.5, green')
pl.plot(circ_1['phir'], circ_1['Circularity_g'],
        marker='x', color='green',
        label='phip=1.0, green')
pl.plot(circ_05['phir'], circ_05['Circularity_r'],
        marker='o', color='red',
        label='phip=0.5, red')
pl.plot(circ_1['phir'], circ_1['Circularity_r'],
        marker='x', color='red',
        label='phip=1.0, red')
pl.plot(circ_05['phir'], circ_05['Circularity_sum'],
        marker='o', color='black',
        label='phip=0.5, red+green')
pl.plot(circ_1['phir'], circ_1['Circularity_sum'],
        marker='x', color='black',
        label='phip=0.5, red+green')
pl.xlabel('phi_r')
pl.ylabel('Circularity')
pl.title('Preliminary circularity plot')
pl.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
time = datetime.now().strftime('%Y-%m-%d_%H%M')
saveFile = '/Users/s1101153/Dropbox/Emily/Graphs/circularity/'+time
pl.savefig(saveFile)
pl.show()
