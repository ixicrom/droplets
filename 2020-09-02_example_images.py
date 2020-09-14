from full_analysis_tools import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import pyplot as pl

infoFileA = '/Users/s1101153/Desktop/droplet_stacks/63x/stack_info_2020-08-28_A.csv'
# infoFileB='/Users/s1101153/Desktop/droplet_stacks/63x/stack_info_2020-08-28_B.csv'
imagePath = '/Users/s1101153/Desktop/droplet_stacks/63x/final_images/ims_to_read/'

# %% read previously calculated slices

rect_data_A = read_rectangle_folder('/Users/s1101153/Desktop/droplet_stacks/63x/final_images/rectangle_slices/A/')
# rect_data_B = read_rectangle_folder('/Users/s1101153/Desktop/droplet_stacks/63x/final_images/rectangle_slices/B/')

# %% format data and make sure images are on the same scale with values 0-1

# rect_dat_A = format_rectangles(rect_data_A, scale='minmax', theta_av=False)
# rect_dat_B = format_rectangles(rect_data_B, scale='minmax', theta_av=False)

r_dat_A = format_rectangles(rect_data_A, scale='minmax', theta_av=True)
# r_dat_B = format_rectangles(rect_data_B, scale='minmax', theta_av=True)


# %% calculate wedge variables from slices already calculated
wedge_dat_A = read_calc_format_wedges(scale='minmax', fileName='/Users/s1101153/Desktop/droplet_stacks/63x/final_images/wedge_slices_A_minmax.pkl', reslice=False, hp=False)

wedge_dat_B = read_calc_format_wedges(scale='minmax', fileName='/Users/s1101153/Desktop/droplet_stacks/63x/final_images/wedge_slices_B_minmax.pkl', reslice=False, hp=False)


# %% combine wedge and theta-averaged data

combi_dat_A = pd.concat([r_dat_A, wedge_dat_A], sort=False, axis=1).dropna()
# combi_dat_B = pd.concat([r_dat_B, wedge_dat_B], sort=False, axis=1).dropna()

scaler = StandardScaler()
combi_dat_A_scaled = pd.DataFrame(scaler.fit_transform(combi_dat_A),
                                  index=combi_dat_A.index,
                                  columns=combi_dat_A.columns)

# combi_dat_B_scaled = pd.DataFrame(scaler.fit_transform(combi_dat_B), index=combi_dat_B.index, columns=combi_dat_B.columns)


combi_dat_A = combi_dat_A_scaled
# combi_dat_B = combi_dat_B_scaled

# r_dat_A_scaled = pd.DataFrame(scaler.fit_transform(r_dat_A), index=r_dat_A.index, columns=r_dat_A.columns)
# r_dat_A = r_dat_A_scaled
# %% visualise the images I want
idx = pd.IndexSlice
dat25_g = r_dat_A.loc[idx['phip0-5_phir25', 'green'], :]
dat25_r = r_dat_A.loc[idx['phip0-5_phir25', 'red'], :]

dat75_g = r_dat_A.loc[idx['phip0-5_phir75', 'green'], :]
dat75_r = r_dat_A.loc[idx['phip0-5_phir75', 'red'], :]

dat40_g = r_dat_A.loc[idx['phip0-5_phir40', 'green'], :]
dat40_r = r_dat_A.loc[idx['phip0-5_phir40', 'red'], :]


# pl.subplot(3,2,1)
# pl.imshow(dat25_r.values)
# pl.title('phi_p=0.5, phi_r=25, red')
# pl.xlabel('r (pixels)')
# pl.ylabel('slice')
#
# pl.subplot(3,2,2)
# pl.imshow(dat75_r.values)
# pl.title('phi_p=0.5, phi_r=75, red')
# pl.xlabel('r (pixels)')
# pl.ylabel('slice')
#
# pl.subplot(3,2,3)
# pl.imshow(dat25_g.values)
# pl.title('phi_p=0.5, phi_r=25, green')
# pl.xlabel('r (pixels)')
# pl.ylabel('slice')
#
# pl.subplot(3,2,4)
# pl.imshow(dat75_g.values)
# pl.title('phi_p=0.5, phi_r=75, green')
# pl.xlabel('r (pixels)')
# pl.ylabel('slice')
#
# pl.subplot(3,2,5)
# pl.imshow(dat40_r.values)
# pl.title('phi_p=0.5, phi_r=40, red')
# pl.xlabel('r (pixels)')
# pl.ylabel('slice')
#
# pl.subplot(3,2,6)
# pl.imshow(dat40_g.values)
# pl.title('phi_p=0.5, phi_r=40, green')
# pl.xlabel('r (pixels)')
# pl.ylabel('slice')
#
# pl.show()


# %% see which slices are in which clusters

# r_dat_A_new = r_dat_A[r_dat_A.index.isin(wedge_dat_A.index)]

# r_dat_A = r_dat_A_new

k_labs, k_count = kmeans(r_dat_A, 5, 'k-means clusters')
# bar_stack(k_count)

labs25_r = k_labs.loc[idx['phip0-5_phir25', 'red'], :]
labs25_g = k_labs.loc[idx['phip0-5_phir25', 'green'], :]

labs75_r = k_labs.loc[idx['phip0-5_phir75', 'red'], :]
labs75_g = k_labs.loc[idx['phip0-5_phir75', 'green'], :]

labs40_r = k_labs.loc[idx['phip0-5_phir40', 'red'], :]
labs40_g = k_labs.loc[idx['phip0-5_phir40', 'green'], :]


# %%


fig, axs = pl.subplots(3, 2, sharex=True, sharey=True)
axs[0, 0].set_ylabel('slice')
axs[0, 0].imshow(dat25_r)
for i in range(dat25_r.shape[0]):
    for j in range(dat25_r.shape[1]):
        text = axs[0, 0].text(j, i, labs25_r.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[0, 0].set_title('phip=0.5, phir=25, red')

axs[1, 0].set_ylabel('slice')
axs[1, 0].imshow(dat25_g)
for i in range(dat25_g.shape[0]):
    for j in range(dat25_g.shape[1]):
        text = axs[1, 0].text(j, i, labs25_g.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[1, 0].set_title('phip=0.5, phir=25, green')

axs[0, 1].imshow(dat75_r)
for i in range(dat75_r.shape[0]):
    for j in range(dat75_r.shape[1]):
        text = axs[0, 1].text(j, i, labs75_r.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[0, 1].set_title('phip=0.5, phir=75, red')

axs[1, 1].imshow(dat75_g)
for i in range(dat75_g.shape[0]):
    for j in range(dat75_g.shape[1]):
        text = axs[1, 1].text(j, i, labs75_g.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[1, 1].set_title('phip=0.5, phir=75, green')

axs[2, 0].set_ylabel('slice')
axs[2, 0].set_xlabel('r (pixels)')
axs[2, 0].imshow(dat40_r)
for i in range(dat40_r.shape[0]):
    for j in range(dat40_r.shape[1]):
        text = axs[2, 0].text(j, i, labs40_r.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[2, 0].set_title('phip=0.5, phir=40, red')

axs[2, 1].set_xlabel('r (pixels)')
axs[2, 1].imshow(dat40_g)
for i in range(dat40_g.shape[0]):
    for j in range(dat40_g.shape[1]):
        text = axs[2, 1].text(j, i, labs40_g.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[2, 1].set_title('phip=0.5, phir=40, green')

fig.suptitle('Standard scaled theta-averaged data')
pl.show()


# %% same but with combi data
idx = pd.IndexSlice
combi_dat25_g = combi_dat_A.loc[idx['phip0-5_phir25', 'green'], :]
combi_dat25_r = combi_dat_A.loc[idx['phip0-5_phir25', 'red'], :]

combi_dat75_g = combi_dat_A.loc[idx['phip0-5_phir75', 'green'], :]
combi_dat75_r = combi_dat_A.loc[idx['phip0-5_phir75', 'red'], :]

combi_dat40_g = combi_dat_A.loc[idx['phip0-5_phir40', 'green'], :]
combi_dat40_r = combi_dat_A.loc[idx['phip0-5_phir40', 'red'], :]


combi_k_labs, combi_k_count = kmeans(combi_dat_A, 5, 'k-means clusters')
# bar_stack(k_count)

combi_labs25_r = combi_k_labs.loc[idx['phip0-5_phir25', 'red'], :]
combi_labs25_g = combi_k_labs.loc[idx['phip0-5_phir25', 'green'], :]

combi_labs75_r = combi_k_labs.loc[idx['phip0-5_phir75', 'red'], :]
combi_labs75_g = combi_k_labs.loc[idx['phip0-5_phir75', 'green'], :]

combi_labs40_r = combi_k_labs.loc[idx['phip0-5_phir40', 'red'], :]
combi_labs40_g = combi_k_labs.loc[idx['phip0-5_phir40', 'green'], :]


# %% plot for the combi data

fig, axs = pl.subplots(3, 2, sharex=True, sharey=True)
axs[0, 0].set_ylabel('slice')
axs[0, 0].imshow(combi_dat25_r)
for i in range(combi_dat25_r.shape[0]):
    for j in range(combi_dat25_r.shape[1]):
        text = axs[0, 0].text(j, i, combi_labs25_r.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[0, 0].set_title('phip=0.5, phir=25, red')

axs[1, 0].set_ylabel('slice')
axs[1, 0].imshow(combi_dat25_g)
for i in range(combi_dat25_g.shape[0]):
    for j in range(combi_dat25_g.shape[1]):
        text = axs[1, 0].text(j, i, combi_labs25_g.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[1, 0].set_title('phip=0.5, phir=25, green')

axs[0, 1].imshow(combi_dat75_r)
for i in range(combi_dat75_r.shape[0]):
    for j in range(combi_dat75_r.shape[1]):
        text = axs[0, 1].text(j, i, combi_labs75_r.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[0, 1].set_title('phip=0.5, phir=75, red')

axs[1, 1].imshow(combi_dat75_g)
for i in range(combi_dat75_g.shape[0]):
    for j in range(combi_dat75_g.shape[1]):
        text = axs[1, 1].text(j, i, combi_labs75_g.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[1, 1].set_title('phip=0.5, phir=75, green')

axs[2, 0].set_ylabel('slice')
axs[2, 0].set_xlabel('r (pixels)')
axs[2, 0].imshow(combi_dat40_r)
for i in range(combi_dat40_r.shape[0]):
    for j in range(combi_dat40_r.shape[1]):
        text = axs[2, 0].text(j, i, combi_labs40_r.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[2, 0].set_title('phip=0.5, phir=40, red')

axs[2, 1].set_xlabel('r (pixels)')
axs[2, 1].imshow(combi_dat40_g)
for i in range(combi_dat40_g.shape[0]):
    for j in range(combi_dat40_g.shape[1]):
        text = axs[2, 1].text(j, i, combi_labs40_g.loc[i].values[0][0],
                              ha='center', va='center',
                              color='w', size=5)
axs[2, 1].set_title('phip=0.5, phir=40, green')
fig.suptitle('Combination data')
pl.show()

# pd.DataFrame(scaler.fit_transform(combi_dat_A)).describe()
# combi_dat25_g.describe()
# dat25_g.describe()
