from full_analysis_tools import read_files, format_rectangles, hier, gini_score, read_calc_format_wedges
from data_tools import norm_data
import matplotlib.pyplot as pl
import os
import numpy as np
import pandas as pd
import matplotlib
import glob
matplotlib.style.core.reload_library()
pl.style.use('thesis')

graph_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP/Graphs/how_clustered/micron_scale/'

# %% rectangular/theta-averaged data
filePath = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/rect_pickles'
filePath_um = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/rect_pickles_final'
imagePath = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/final_images/ims_to_read/'

dat2 = read_files(filePath)
dat_list = []
for file in glob.glob(os.path.join(filePath_um,'*.pkl')):
    dat_list.append(pd.read_pickle(file))
dat = pd.concat(dat_list, axis=1)
dat.shape
dat_scaled = norm_data(dat)
dat = dat_scaled.dropna()
dat.describe()
dat.columns[0]
dat2.columns[1]

file_suffix = ''
r_dat = format_rectangles(dat, scale='standard', theta_av=True, rCol=dat.columns[0])
file_suffix += '_theta-av'

wedge_path = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/'


info_file = os.path.join(wedge_path, 'stack_info.csv')
save_file = os.path.join(wedge_path, 'wedges_all')+'.pkl'
wedges = read_calc_format_wedges(scale='standard',
                                 fileName=save_file,
                                 reslice=True,
                                 imPath=imagePath,
                                 infoFile=info_file,
                                 hp=False)
r_dat = r_dat[r_dat.index.isin(wedges.index)]
r_dat.shape
r_dat.describe()
# %% calculate wedge data
# wedge_path = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/'
#
# info_file = os.path.join(wedge_path, 'stack_info.csv')
# save_file = os.path.join(wedge_path, 'wedges_all')+'.pkl'
# file_suffix += '_all'
#
# wedges = read_calc_format_wedges(scale='minmax',
#                                  fileName=save_file,
#                                  reslice=True,
#                                  imPath=imagePath,
#                                  infoFile=info_file,
#                                  hp=False)
#
# r_dat = r_dat[r_dat.index.isin(wedges.index)]


# %% hierarchical clustering

dendro = False
h_labs, h_count = hier(r_dat, 5, show_dendro=dendro)
h_gini = gini_score(h_count)
print('hier_gini'+file_suffix+': '+str(np.mean(h_gini)))

h_count.transpose().plot(kind='bar', stacked=True)
pl.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
pl.xlabel('Original image')
pl.ylabel('Cluster frequency')
pl.title('Hierarchical clusters'+file_suffix.replace('_', ', '))
pl.tight_layout()
# pl.savefig(graph_folder+'hier_bar'+file_suffix+'.png')
pl.show()

idx = pd.IndexSlice
max_val = 1  # r_dat.max().max()
min_val = 0  # r_dat.min().min()
samples = r_dat.index.get_level_values(0).unique()
asp = r_dat.shape[1]*5/200
s = samples[0]
for s in samples:
    dat_g = r_dat.loc[idx[s, 'green'], :]
    dat_r = r_dat.loc[idx[s, 'red'], :]
    labs_g = h_labs.loc[idx[s, 'green'], :]
    labs_r = h_labs.loc[idx[s, 'red'], :]

    s_title = s.replace('-', '.').replace('\_', ', ').replace('phip', 'phi\_p=').replace('phir', 'phi\_r=')
    if dat_r.shape[0] == 12:
        fig, ax = pl.subplots()
        ax.set_aspect('equal')
        ax.set_ylabel('Slice')
        ax.set_xlabel('r (0.1$\mu$m)')
        ax.imshow(dat_r, aspect=asp, vmin=min_val, vmax=max_val)

        for i in range(dat_r.shape[0]):
            for j in range(dat_r.shape[1]):
                if j % 10 == 0:
                    text = ax.text(j, i, labs_r.loc[i].values[0][0],
                                       ha='center', va='center',
                                       color='w', size=7)
        # ax.set_title(s_title + '\n Red channel')
        pl.savefig(graph_folder+'scaled_'+s.replace('\_', '_')+'_red_no-title.png')
        pl.close()

    if dat_g.shape[0] == 12:
        fig, ax = pl.subplots()
        ax.set_aspect('equal')
        ax.set_ylabel('Slice')
        ax.set_xlabel('r (0.1$\mu$m)')
        ax.imshow(dat_g, aspect=asp, vmin=min_val, vmax=max_val)

        for i in range(dat_g.shape[0]):
            for j in range(dat_g.shape[1]):
                if j % 10 == 0:
                    text = ax.text(j, i, labs_g.loc[i].values[0][0],
                                       ha='center', va='center',
                                       color='w', size=7)
        # ax.set_title(s_title + '\n Green channel')
        pl.savefig(graph_folder+'scaled_'+s.replace('\_', '_')+'_green_no-title.png')
        pl.close()
dat_g.describe()
