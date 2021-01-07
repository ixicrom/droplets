from full_analysis_tools import slice_info_file_um, read_files
import pandas as pd
import os
import glob

save_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/rect_pickles_um/'

info_file = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/stack_info.csv'

image_path = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/final_images/ims_to_read/'

slices = slice_info_file_um(info_file, image_path, 12, 100, savePath=save_folder)

read_files('/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/rect_pickles/')
r_dat = format_rectangles(dat, scale='standard', theta_av=False)
print(r_dat.head())


search = os.path.join(save_folder, "*.pkl")
file_names = glob.glob(search)
slice_list = []
for file in file_names:
    slice_list.append(pd.read_pickle(file))

slice_df = pd.concat(slice_list, axis=1)
slice_df.dropna()
