from full_analysis_tools import read_files, format_rectangles, PCA_transform
import matplotlib.pyplot as pl
import pandas as pd
import numpy as np

graph_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP/Graphs/final_for_thesis/'

# %% rectangular/theta-averaged data
filePath = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/rect_pickles'
imagePath = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/final_images/ims_to_read/'

dat = read_files(filePath)
dat.head()

r_dat = format_rectangles(dat, scale='standard', theta_av=False)
r_dat.head()
pca99, pca_dat99 = PCA_transform(r_dat, 0.99)

pca97, pca_dat97 = PCA_transform(r_dat, 0.97)

pca_dat97.head()
pca_difference = pca_dat99.loc[:,186:]
pca_difference

pca_extra_comps = pd.DataFrame(pca99.components_)[186:]

r = pd.Series(np.sort([*range(273)]*100), name='r')
theta = pd.Series([*range(100)]*273, name='theta')
pca_extra_comps = pca_extra_comps.append(theta)
pca_extra_comps = pca_extra_comps.append(r)

pca_extra_comps_toplot = pca_extra_comps.T.set_index(['theta', 'r'])

pca_extra_comps_toplot

for i in pca_extra_comps_toplot.columns:
    pl.pcolor(pca_extra_comps_toplot[i].unstack())
    pl.title('Principal component '+str(i)+'/265')
    pl.xlabel('r (pixels)')
    pl.ylabel('theta (% of slice)')
    pl.show()
