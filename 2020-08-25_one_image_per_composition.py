from full_analysis_tools import *
import pandas as pd



# %% rectangular data
filePath = '/Users/s1101153/Desktop/droplet_stacks/63x/rect_pickles'
imagePath = '/Users/s1101153/Desktop/droplet_stacks/63x/final_images/ims_to_read/'

r_dat = read_format_rectangles(filePath=filePath, scale='standard', theta_av=True)

rect_dat = read_format_rectangles(filePath=filePath, scale='standard', theta_av=False)


# %% wedge data
infoFileOld = '/Users/s1101153/Desktop/droplet_stacks/63x/stack_info.csv'

infoFile = '/Users/s1101153/Desktop/droplet_stacks/63x/stack_info_2020-08-25.csv'

readFile = '/Users/s1101153/Desktop/droplet_stacks/63x/nice_slices.pkl'
saveFile = '/Users/s1101153/Desktop/droplet_stacks/63x/nice_slices_2020-08-25.pkl'


old_wedges = read_calc_format_wedges(scale='standard', fileName = readFile, reslice=False)
old_wedges

# old_wedges.reset_index()

new_wedges = read_calc_format_wedges(scale='standard', fileName = saveFile, reslice = True, imPath = imagePath, infoFile = infoFile, hp=False)
new_wedges


# %% combined data
combi_dat = pd.concat([rect_dat, new_wedges], sort=False, axis=1)

combi_dat = combi_dat.dropna()

# %% rectangular data for just new images
new_rect_dat = rect_dat[rect_dat.index.isin(new_wedges.index)]
new_r_dat = r_dat[r_dat.index.isin(new_wedges.index)]


# %% gini score Plots

r_score_old = gini_score_range(method='k', dat=r_dat, start_num=3, stop_num=33)
r_score_new = gini_score_range(method='k', dat=new_r_dat, start_num=3, stop_num=33)

x=[*range(3,33)]
pl.plot(x, r_score_old, label='All images, theta-average')
pl.plot(x, r_score_new, label='Max. 1 image per composition, theta-average')
pl.legend()
pl.xlabel('Number of clusters')
pl.ylabel('Gini score')
pl.title("K-means Gini Scores (standard scaled)")
pl.show()


#  %% t-SNE visualisation

perp_vals = [5,10,30,50,100]

for p in perp_vals:
    dat_tsne_test = tSNE_transform(new_rect_dat, p)
    pl.scatter(dat_tsne_test[:,0], dat_tsne_test[:,1])
    pl.title('t-SNE (standard scaled, 1 image per composition), p = ' +str(p))
    pl.show()




# %% look at which images we have in the p=5 case for this data

dat_tsne_new = tSNE_transform(new_rect_dat, 5)
sample_names=new_rect_dat.index.get_level_values(0)

tSNE_plot(dat_tsne_new, sample_names, 'New t-SNE, standard scaled, coloured by sample name', alpha=0.7)

#  %% do k-means clustering and make plots

k_labs, k_count = kmeans(new_r_dat, 4, 'k-means theta-averaged')

bar_stack(k_count)

phi_plot(k_labs, 'k-means theta-averaged', 'K-means new, 4 clusters (theta-averaged, standard scaled)')


#  %% do hierarchical clustering and make plots

h_labs, h_count = kmeans(new_r_dat, 4, 'hierarchical theta-averaged')

bar_stack(h_count)

phi_plot(h_labs, 'hierarchical theta-averaged', 'Hierarchical new, 4 clusters (theta-averaged, standard scaled)')



# %% play with inertia plot for k-means

inertia_vals=[]
for n_clust in range(3,15):
    np.random.seed(1234)
    kmeans=cluster.KMeans(n_clusters=n_clust)
    kmeans_fit=kmeans.fit(new_r_dat)
    inertia_vals.append(kmeans_fit.inertia_)

pl.plot([*range(3,15)], inertia_vals)
pl.xlabel('Number of clusters')
pl.ylabel('K-means inertia')
pl.vlines([4,7], ymin=12000, ymax=18500, linestyle='dashed')
pl.show()

# %%


import kneed

kn=kneed.KneeLocator([*range(3,15)], inertia_vals, curve='convex', direction='decreasing', interp_method='polynomial')

print(kn.knee)

kn_gini = kneed.KneeLocator([*range(3,33)], r_score_new, curve='convex', direction='decreasing', interp_method='polynomial')
print(kn_gini.knee)

kn_gini_old = kneed.KneeLocator([*range(3,33)], r_score_old, curve='convex', direction='decreasing', interp_method='polynomial')
print(kn_gini_old.knee)
