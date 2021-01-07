from full_analysis_tools import read_files, format_rectangles, read_calc_format_wedges, hier, clust, gini_score_range, optimalK, tSNE_transform, gini_score, phi_plot, PCA_transform, tSNE_transform, tSNE_plot_2col
import matplotlib.pyplot as pl
import os
import numpy as np
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.style.core.reload_library()
pl.style.use('thesis')

graph_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP/Graphs/how_clustered/'

# %% rectangular/theta-averaged data
filePath = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/rect_pickles'
imagePath = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/final_images/ims_to_read/'

dat = read_files(filePath)
file_suffix = ''
if input('Use theta-averaged data? y/n: ') == 'y':
    r_dat = format_rectangles(dat, scale='standard', theta_av=True)
    file_suffix += '_theta-av'
elif input('Average over r instead? y/n: ') == 'y':
    r_dat = format_rectangles(dat, scale='standard', theta_av=False, r_av=True)
    file_suffix += '_r-av'
else:
    r_dat = format_rectangles(dat, scale='standard', theta_av=False, r_av=False)
    file_suffix += '_rect'

# %% calculate wedge data
wedge_path = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/'

data_tag = input('Which data to include? all/nice/oneA/oneB: ')
if data_tag == 'all':
    info_file = os.path.join(wedge_path, 'stack_info.csv')
    save_file = os.path.join(wedge_path, 'wedges_all')+'.pkl'
    file_suffix += '_all'
elif data_tag == 'nice':
    info_file = os.path.join(wedge_path, 'stack_info_2020-08-20.csv')
    save_file = os.path.join(wedge_path, 'wedges_nice')+'.pkl'
    file_suffix += '_nice'
elif data_tag == 'oneA':
    info_file = os.path.join(wedge_path, 'stack_info_2020-08-28_A.csv')
    save_file = os.path.join(wedge_path, 'wedges_A.pkl')
    file_suffix += '_oneA'
elif data_tag == 'oneB':
    info_file = os.path.join(wedge_path, 'stack_info_2020-08-28_B.csv')
    save_file = os.path.join(wedge_path, 'wedges_B')+'.pkl'
    file_suffix += '_oneB'
elif data_tag == 'noDist':
    info_file = os.path.join(wedge_path, 'stack_info_no_distorted.csv')
    save_file = os.path.join(wedge_path, 'wedges_noDist')+'.pkl'
    file_suffix += '_noDist'
elif data_tag == 'noIsh':
    info_file = os.path.join(wedge_path, 'stack_info_no_roundish.csv')
    save_file = os.path.join(wedge_path, 'wedges_noIsh')+'.pkl'
    file_suffix += '_noIsh'
else:
    print('Invalid selection, problems will happen')

wedges = read_calc_format_wedges(scale='minmax',
                                 fileName=save_file,
                                 reslice=True,
                                 imPath=imagePath,
                                 infoFile=info_file,
                                 hp=False)

r_dat = r_dat[r_dat.index.isin(wedges.index)]

# %% PCA
if input('Use PCA? y/n: ') == 'y':
    var = float(input('PCA variance: '))
    file_suffix += '_pca'+str(round(var*100))
    r_dat = PCA_transform(r_dat, var)[1]


r_dat.shape
5/200

# %% hierarchical clustering

dendro = False
h_labs, h_count = hier(r_dat, 5, show_dendro=dendro)
h_gini = gini_score(h_count)
print('hier_gini'+file_suffix+': '+str(np.mean(h_gini)))
h_count.transpose().plot(kind='bar', stacked=True, figsize=(7, 10.5))#todo: make plot shorter or wider and/or axis labels smaller
pl.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
pl.xlabel('Original image')
pl.ylabel('Cluster frequency')
pl.title('Hierarchical clusters'+file_suffix.replace('_', '\_'))
pl.tight_layout()
pl.savefig(graph_folder+'hier_bar'+file_suffix+'.png')
pl.show()

# %% plot example images set 1
idx = pd.IndexSlice
dat25_g = r_dat.loc[idx['phip0-5\_phir25', 'green'], :]
dat25_r = r_dat.loc[idx['phip0-5\_phir25', 'red'], :]

dat75_g = r_dat.loc[idx['phip0-5\_phir75', 'green'], :]
dat75_r = r_dat.loc[idx['phip0-5\_phir75', 'red'], :]

dat40_g = r_dat.loc[idx['phip0-5\_phir40', 'green'], :]
dat40_r = r_dat.loc[idx['phip0-5\_phir40', 'red'], :]


# now get the cluster labels
labs25_r = h_labs.loc[idx['phip0-5\_phir25', 'red'], :]

labs25_g = h_labs.loc[idx['phip0-5\_phir25', 'green'], :]

labs75_r = h_labs.loc[idx['phip0-5\_phir75', 'red'], :]
labs75_g = h_labs.loc[idx['phip0-5\_phir75', 'green'], :]

labs40_r = h_labs.loc[idx['phip0-5\_phir40', 'red'], :]
labs40_g = h_labs.loc[idx['phip0-5\_phir40', 'green'], :]

asp = r_dat.shape[1]*5/200
# now see if the plot works!
fig, axs = pl.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 10))
axs[0, 0].set_aspect('equal')
axs[0, 0].set_ylabel('slice')
axs[0, 0].imshow(dat25_r, aspect=asp)
for i in range(dat25_r.shape[0]):

    for j in range(dat25_r.shape[1]):
        if j % 10 == 0:
            text = axs[0, 0].text(j, i, labs25_r.loc[i].values[0][0],
                                  ha='center', va='center',
                                  color='w', size=7)
axs[0, 0].set_title('phip=0.5, phir=25, red')

axs[1, 0].set_ylabel('slice')
axs[1, 0].imshow(dat25_g, aspect=asp)
for i in range(dat25_g.shape[0]):
    for j in range(dat25_g.shape[1]):
        if j % 10 == 0:
            text = axs[1, 0].text(j, i, labs25_g.loc[i].values[0][0],
                                  ha='center', va='center',
                                  color='w', size=7)
axs[1, 0].set_title('phip=0.5, phir=25, green')

axs[0, 1].imshow(dat75_r, aspect=asp)
for i in range(dat75_r.shape[0]):
    for j in range(dat75_r.shape[1]):
        if j % 10 == 0:
            text = axs[0, 1].text(j, i, labs75_r.loc[i].values[0][0],
                                  ha='center', va='center',
                                  color='w', size=7)
axs[0, 1].set_title('phip=0.5, phir=75, red')

axs[1, 1].imshow(dat75_g, aspect=asp)
for i in range(dat75_g.shape[0]):
    for j in range(dat75_g.shape[1]):
        if j % 10 == 0:
            text = axs[1, 1].text(j, i, labs75_g.loc[i].values[0][0],
                                  ha='center', va='center',
                                  color='w', size=7)
axs[1, 1].set_title('phip=0.5, phir=75, green')

axs[2, 0].set_ylabel('slice')
axs[2, 0].set_xlabel('PCA component')
axs[2, 0].imshow(dat40_r, aspect=asp)
for i in range(dat40_r.shape[0]):
    for j in range(dat40_r.shape[1]):
        if j % 10 == 0:
            text = axs[2, 0].text(j, i, labs40_r.loc[i].values[0][0],
                                  ha='center', va='center',
                                  color='w', size=7)
axs[2, 0].set_title('phip=0.5, phir=40, red')

axs[2, 1].set_xlabel('PCA component')
axs[2, 1].imshow(dat40_g, aspect=asp)
for i in range(dat40_g.shape[0]):
    for j in range(dat40_g.shape[1]):
        if j % 10 == 0:
            text = axs[2, 1].text(j, i,
                                  labs40_g.loc[i].values[0][0],
                                  ha='center', va='center',
                                  color='w', size=7)
axs[2, 1].set_title('phip=0.5, phir=40, green')

fig.suptitle('Examples of input data  '+file_suffix[1:].replace('_', ', '))
fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0)
pl.tight_layout()
pl.savefig(graph_folder+'data_eg'+file_suffix+'_1.png')
pl.show()


# %% plot example images set 2

dat25_g = r_dat.loc[idx['phip0-5\_phir25', 'green'], :]
dat25_r = r_dat.loc[idx['phip0-5\_phir25', 'red'], :]

dat40_g = r_dat.loc[idx['phip0-5\_phir40', 'green'], :]
dat40_r = r_dat.loc[idx['phip0-5\_phir40', 'red'], :]

dat60_g = r_dat.loc[idx['phip0-5\_phir60', 'green'], :]
dat60_r = r_dat.loc[idx['phip0-5\_phir60', 'red'], :]


# now get the cluster labels
labs25_r = h_labs.loc[idx['phip0-5\_phir25', 'red'], :]
labs25_g = h_labs.loc[idx['phip0-5\_phir25', 'green'], :]

labs40_r = h_labs.loc[idx['phip0-5\_phir40', 'red'], :]
labs40_g = h_labs.loc[idx['phip0-5\_phir40', 'green'], :]

labs60_r = h_labs.loc[idx['phip0-5\_phir60', 'red'], :]
labs60_g = h_labs.loc[idx['phip0-5\_phir60', 'green'], :]


# now see if the plot works!
fig, axs = pl.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 10))
axs[0, 0].set_aspect('equal')
axs[0, 0].set_ylabel('slice')

axs[0, 0].imshow(dat25_r, aspect=asp)
for i in range(dat25_r.shape[0]):

    for j in range(dat25_r.shape[1]):
        if j % 10 == 0:
            text = axs[0, 0].text(j, i, labs25_r.loc[i].values[0][0],
                                  ha='center', va='center',
                                  color='w', size=7)
axs[0, 0].set_title('phip=0.5, phir=25, red')

axs[1, 0].set_ylabel('slice')

axs[1, 0].imshow(dat25_g, aspect=asp)
for i in range(dat25_g.shape[0]):
    for j in range(dat25_g.shape[1]):
        if j % 10 == 0:
            text = axs[1, 0].text(j, i, labs25_g.loc[i].values[0][0],
                                  ha='center', va='center',
                                  color='w', size=7)
axs[1, 0].set_title('phip=0.5, phir=25, green')

axs[0, 1].imshow(dat40_r, aspect=asp)
for i in range(dat40_r.shape[0]):
    for j in range(dat40_r.shape[1]):
        if j % 10 == 0:
            text = axs[0, 1].text(j, i, labs40_r.loc[i].values[0][0],
                                  ha='center', va='center',
                                  color='w', size=7)
axs[0, 1].set_title('phip=0.5, phir=40, red')

axs[1, 1].imshow(dat60_g, aspect=asp)
for i in range(dat60_g.shape[0]):
    for j in range(dat60_g.shape[1]):
        if j % 10 == 0:
            text = axs[1, 1].text(j, i, labs60_g.loc[i].values[0][0],
                                  ha='center', va='center',
                                  color='w', size=7)
axs[1, 1].set_title('phip=0.5, phir=60, green')

axs[2, 0].set_ylabel('slice')
axs[2, 0].set_xlabel('PCA component')

axs[2, 0].imshow(dat60_r, aspect=asp)
for i in range(dat60_r.shape[0]):
    for j in range(dat60_r.shape[1]):
        if j % 10 == 0:
            text = axs[2, 0].text(j, i, labs60_r.loc[i].values[0][0],
                                  ha='center', va='center',
                                  color='w', size=7)
axs[2, 0].set_title('phip=0.5, phir=60, red')

axs[2, 1].set_xlabel('PCA component')
axs[2, 1].imshow(dat40_g, aspect=asp)
for i in range(dat40_g.shape[0]):
    for j in range(dat40_g.shape[1]):
        if j % 10 == 0:
            text = axs[2, 1].text(j, i,
                                  labs40_g.loc[i].values[0][0],
                                  ha='center', va='center',
                                  color='w', size=7)
axs[2, 1].set_title('phip=0.5, phir=40, green')

fig.suptitle('Examples of input data  '+file_suffix[1:].replace('_', ', '))
fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0)
pl.tight_layout()
pl.savefig(graph_folder+'data_eg'+file_suffix+'_2.png')
pl.show()
