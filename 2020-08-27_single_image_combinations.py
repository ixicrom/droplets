from full_analysis_tools import *
import pandas as pd

infoFileA='/Users/s1101153/Desktop/droplet_stacks/63x/stack_info_2020-08-20.csv'
imagePath='/Users/s1101153/Desktop/droplet_stacks/63x/final_images/ims_to_read/'

# rect_data = slice_info_file(infoFile=infoFileA, imPath=imagePath, n_slice=12, n_theta=100, savePath='/Users/s1101153/Desktop/droplet_stacks/63x/final_images/rectangle_slices/')

# file_dat = read_files('/Users/s1101153/Desktop/droplet_stacks/63x/rect_pickles')

# test2 = theta_average(file_dat).transpose()
# samp2=test2.reset_index()['slices'].str.slice(4,21).str.rstrip('_stack')
# test = theta_average(rect_data).transpose()
# samp=test.reset_index()['slices'].str.slice(4,21).str.rstrip('_stack')
#
# samp2
# samp


rect_data = read_rectangle_folder('/Users/s1101153/Desktop/droplet_stacks/63x/final_images/rectangle_slices/')

rect_dat = format_rectangles(rect_data, 'standard', theta_av=False)


r_dat = format_rectangles(rect_data, 'standard', theta_av=True)
