from full_analysis_tools import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

infoFileA='/Users/s1101153/Desktop/droplet_stacks/63x/stack_info_2020-08-28_A.csv'
infoFileB='/Users/s1101153/Desktop/droplet_stacks/63x/stack_info_2020-08-28_B.csv'
imagePath='/Users/s1101153/Desktop/droplet_stacks/63x/final_images/ims_to_read/'

# %% if calculating slices from infoFiles

# rect_data_A = slice_info_file(infoFile=infoFileA, imPath=imagePath, n_slice=12, n_theta=100, r_min=191, r_max=290, savePath='/Users/s1101153/Desktop/droplet_stacks/63x/final_images/rectangle_slices/A/')
# rect_data_B = slice_info_file(infoFile=infoFileB, imPath=imagePath, n_slice=12, n_theta=100, r_min=191, r_max=304, savePath='/Users/s1101153/Desktop/droplet_stacks/63x/final_images/rectangle_slices/B/')


# %% if reading previously calculated slices

rect_data_A = read_rectangle_folder('/Users/s1101153/Desktop/droplet_stacks/63x/final_images/rectangle_slices/A/')
rect_data_B = read_rectangle_folder('/Users/s1101153/Desktop/droplet_stacks/63x/final_images/rectangle_slices/B/')

# %% format data and make sure images are on the same scale with values 0-1

rect_dat_A = format_rectangles(rect_data_A, scale='minmax', theta_av=False)
rect_dat_B = format_rectangles(rect_data_B, scale='minmax', theta_av=False)

r_dat_A = format_rectangles(rect_data_A, scale='minmax', theta_av=True)
r_dat_B = format_rectangles(rect_data_B, scale='minmax', theta_av=True)


# %% calculate wedge variables from scratch
# wedge_dat_A = read_calc_format_wedges(scale='minmax', fileName='/Users/s1101153/Desktop/droplet_stacks/63x/final_images/wedge_slices_A_minmax.pkl', reslice=True, imPath=imagePath, infoFile=infoFileA, hp=False)
#
# wedge_dat_B = read_calc_format_wedges(scale='minmax', fileName='/Users/s1101153/Desktop/droplet_stacks/63x/final_images/wedge_slices_B_minmax.pkl', reslice=True, imPath=imagePath, infoFile=infoFileB, hp=False)
#
#
# wedge_dat_A.index.get_level_values(0).unique()

# %% calculate wedge variables from slices already calculated
wedge_dat_A = read_calc_format_wedges(scale='minmax', fileName='/Users/s1101153/Desktop/droplet_stacks/63x/final_images/wedge_slices_A_minmax.pkl', reslice=False, hp=False)

wedge_dat_B = read_calc_format_wedges(scale='minmax', fileName='/Users/s1101153/Desktop/droplet_stacks/63x/final_images/wedge_slices_B_minmax.pkl', reslice=False, hp=False)


# %% combine wedge and theta-averaged data

combi_dat_A = pd.concat([r_dat_A, wedge_dat_A], sort=False, axis=1).dropna()
combi_dat_B = pd.concat([r_dat_B, wedge_dat_B], sort=False, axis=1).dropna()

scaler = StandardScaler()
combi_dat_A_scaled = pd.DataFrame(scaler.fit_transform(combi_dat_A), index=combi_dat_A.index, columns=combi_dat_A.columns)

combi_dat_B_scaled = pd.DataFrame(scaler.fit_transform(combi_dat_B), index=combi_dat_B.index, columns=combi_dat_B.columns)


combi_dat_A = combi_dat_A_scaled
combi_dat_B = combi_dat_B_scaled

# %% take intersection of wedge and rectangular data

rect_dat_A_new = rect_dat_A[rect_dat_A.index.isin(wedge_dat_A.index)]
rect_dat_B_new = rect_dat_B[rect_dat_B.index.isin(wedge_dat_B.index)]

rect_dat_A = rect_dat_A_new
rect_dat_B = rect_dat_B_new

# %% onto the analysis! first check how many clusters t-SNE suggests

dat_tsne_A = tSNE_transform(rect_dat_A, 10)
pl.scatter(dat_tsne_A[:,0], dat_tsne_A[:,1])
pl.show()

dat_tsne_B = tSNE_transform(rect_dat_B, 10)
pl.scatter(dat_tsne_B[:,0], dat_tsne_B[:,1])
pl.show()


# %% now do k-means clustering, let's use 5 clusters

k_labs_A, k_count_A = kmeans(r_dat_A, 5, 'k-means_A theta-averaged')
bar_stack(k_count_A)
phi_plot(k_labs_A, 'k-means_A theta-averaged', 'K-means, data A')

k_labs_B, k_count_B = kmeans(r_dat_B, 5, 'k-means_B theta-averaged')
bar_stack(k_count_B)
phi_plot(k_labs_B, 'k-means_B theta-averaged', 'K-means, data B')



# %% t-SNE on combi data
combi_tsne_A = tSNE_transform(combi_dat_A, 5)
pl.scatter(combi_tsne_A[:,0], combi_tsne_A[:,1])
pl.show()

combi_tsne_B = tSNE_transform(combi_dat_A, 5)
pl.scatter(combi_tsne_B[:,0], combi_tsne_B[:,1])
pl.show()


# %%

k_labs_combi_A, k_count_combi_A = kmeans(combi_dat_A, 5, 'k-means_A theta-averaged')
bar_stack(k_count_combi_A)
phi_plot(k_labs_combi_A, 'k-means_A theta-averaged', 'K-means, data A')

k_labs_combi_B, k_count_combi_B = kmeans(combi_dat_B, 5, 'k-means_B theta-averaged')
bar_stack(k_count_combi_B)
phi_plot(k_labs_combi_B, 'k-means_B theta-averaged', 'K-means, data B')

# %% gini scores

g_A = np.mean(gini_score(k_count_A))
print(g_A)

g_B = np.mean(gini_score(k_count_B))
print(g_B)

g_combi_A = np.mean(gini_score(k_count_combi_A))
print(g_combi_A)

g_combi_B = np.mean(gini_score(k_count_combi_B))
print(g_combi_B)
