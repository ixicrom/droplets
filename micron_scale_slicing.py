from full_analysis_tools import slice_info_file_um, rectangle_slice_um
from sklearn.preprocessing import MinMaxScaler
from os import path
from skimage import io
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import glob


info_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/stack_info.csv'

image_path = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/final_images/ims_to_read'

filePath_um = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/rect_pickles_final'

n_slice = 12
n_theta = 100


dat = slice_info_file_um(infoFile=info_file, imPath=image_path, n_slice=n_slice, n_theta=n_theta, r_max_um=14.6, savePath=filePath_um)
