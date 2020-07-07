import pandas as pd
import matplotlib.pyplot as pl
from processing_tools import *
import matplotlib.patches as mpatches


file = "/Users/s1101153/Dropbox/Emily/nice_slices.pkl"

dat=pd.read_pickle(file)

idx=pd.IndexSlice

slices=dat.loc[idx["T2M_6_1",:,:]]

# pl.plot(slices.loc[idx["green",0]][0])
# pl.show()
#
# pl.plot(slices.loc[idx["green",1]][0])
# pl.show()
#
# pl.plot(slices.loc[idx["red",0]][0])
# pl.show()
#
# pl.plot(slices.loc[idx["red",1]][0])
# pl.show()


image=slices.loc[idx["red",1]][0]
label_image = threshold_and_label(image)

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
