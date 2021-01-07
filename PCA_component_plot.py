from full_analysis_tools import read_files, format_rectangles, PCA_transform, clust, gini_score_range
import matplotlib.pyplot as pl
import pandas as pd
import numpy as np
import timeit
from sklearn.decomposition import PCA
import matplotlib
matplotlib.style.core.reload_library()
pl.style.use('thesis')

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
pca_difference = pca_dat99.loc[:, 186:]
pca_difference

pca_extra_comps = pd.DataFrame(pca99.components_)[:10]

r = pd.Series(np.sort([*range(273)]*100), name='r')
theta = pd.Series([*range(100)]*273, name='theta')
pca_extra_comps = pca_extra_comps.append(theta)
pca_extra_comps = pca_extra_comps.append(r)

pca_extra_comps_toplot = pca_extra_comps.T.set_index(['theta', 'r'])

# pca_extra_comps_toplot

# for i in pca_extra_comps_toplot.columns:
#     pl.pcolor(pca_extra_comps_toplot[i].unstack())
#     pl.title('Principal component '+str(i)+'/265')
#     pl.xlabel('r (pixels)')
#     pl.ylabel('theta (% of slice)')
#     pl.show()


# %% plot to show how variance changes the image reconstruction
# from PCA_droplets.py
if input('Make example PCA image? y/n: ')=='y':
    variances = [0.99, 0.97, 0.95, 0.92, 0.9, 0.8, 0.7, 0.5]
    n = len(variances)+1
    pl.subplot(1, n, 1)
    pl.imshow(r_dat.iloc[0].to_numpy().reshape(273, 100))
    pl.xlabel('27300 components')
    pl.title('Original image')

    i = 2
    with pl.style.context(['thesis_wide']):
        for var in variances:
            pca = PCA(var)
            transformed = pca.fit_transform(r_dat)
            approximation = pca.inverse_transform(transformed)
            pl.subplot(1, n, i)
            pl.imshow(approximation[0].reshape(273, 100))
            pl.xlabel(str(pca.n_components_) + ' components')
            pl.title(str(var*100) + '% variance')
            i = i+1
    # pl.suptitle('Impact of PCA on an example image')
    pl.show()


# %% plot to show how the time to run clustering changes with PCA

if input('Make time graph? y/n: ')=='y':
    variances_time = [0.99, 0.97, 0.95, 0.92, 0.9, 0.8, 0.7, 0.5]
    h_times = []
    k_times = []
    x = []

    h_time_orig = timeit.repeat(lambda: clust('h', r_dat, 5),
                                repeat=5,
                                number=10)
    h_times.append(min(h_time_orig))

    k_time_orig = timeit.repeat(lambda: clust('k', r_dat, 5),
                                repeat=5,
                                number=10)
    k_times.append(min(k_time_orig))
    x.append(1.0)

    for var in variances_time:
        pca = PCA(var)
        transformed = pca.fit_transform(r_dat)
        approximation = pca.inverse_transform(transformed)
        approx_dat = pd.DataFrame(approximation, index=r_dat.index)
        h_time_PCA = timeit.repeat(lambda: clust('h', approx_dat, 5),
                                   repeat=5,
                                   number=10)
        h_times.append(min(h_time_PCA))
        k_time_PCA = timeit.repeat(lambda: clust('k', approx_dat, 5),
                                   repeat=5,
                                   number=10)
        k_times.append(min(k_time_PCA))
        x.append(var)


    pl.plot(x, h_times, '-o')
    pl.xlabel('Variance retained in PCA')
    pl.ylabel('Time taken for 10 runs')
    pl.title('Hierarchical clustering')
    pl.show()

    pl.plot(x, k_times, '-o')
    pl.xlabel('Variance retained in PCA')
    pl.ylabel('Time taken for 10 runs')
    pl.title('K-means clustering')
    pl.show()


# %% gini score with and without PCA

if input('Make gini score plot? y/n: ') == 'y':
    k_scores = []
    k_errors = []
    h_scores = []
    h_errors = []
    labs = []
    min_clust = 3
    max_clust = 7
    seeds = [1234, 1111, 4321, 4444]

    k_orig_score = []
    for s in seeds:
        k_orig_score.append(gini_score_range('k',
                                             r_dat,
                                             min_clust,
                                             max_clust,
                                             seed=s))
    k_orig_score_arr = np.array(k_orig_score)
    k_scores.append(list(np.mean(k_orig_score_arr, axis=0)))
    k_errors.append(list(np.std(k_orig_score_arr, axis=0)))

    h_orig_score = gini_score_range('h', r_dat, min_clust, max_clust)
    h_scores.append(h_orig_score)
    h_errors.append([0, 0, 0, 0])
    x = [* range(min_clust, max_clust)]
    labs.append('No PCA')

    variances_gini = [0.99, 0.97, 0.5]

    for var in variances_gini:
        pca = PCA(var)
        transformed = pca.fit_transform(r_dat)
        approximation = pca.inverse_transform(transformed)
        approx_dat = pd.DataFrame(approximation, index=r_dat.index)
        k_score_temp = []
        h_score_temp = []
        for s in seeds:
            k_score_temp.append(gini_score_range('k',
                                                 approx_dat,
                                                 min_clust,
                                                 max_clust,
                                                 seed=s))
            h_score_temp.append(gini_score_range('h',
                                                 approx_dat,
                                                 min_clust,
                                                 max_clust,
                                                 seed=s))

        k_score_temp_arr = np.array(k_score_temp)
        h_score_temp_arr = np.array(h_score_temp)

        k_scores.append(list(np.mean(k_score_temp_arr, axis=0)))
        k_errors.append(list(np.std(k_score_temp_arr, axis=0)))
        h_scores.append(list(np.mean(h_score_temp_arr, axis=0)))
        h_errors.append(list(np.std(h_score_temp_arr, axis=0)))

        lab = 'PCA ' + str(var*100) + '% variance'
        # k_scores.append(k_pca_score)
        # h_scores.append(h_pca_score)
        labs.append(lab)

    # for i in range(len(scores)):
    #     pl.plot(x, k_scores[i], '-o', label=labs[i])
    # pl.xlabel('Number of clusters')
    # pl.ylabel('Gini score')
    # pl.legend()
    # pl.show()

    for i in range(len(h_scores)):
        pl.errorbar(x, h_scores[i],
                    xerr=None, yerr=h_errors[i],
                    fmt='-o', capsize=5,
                    label=labs[i])
    pl.xlabel('Number of clusters')
    pl.ylabel('Gini score')
    pl.legend()  # loc='center left', bbox_to_anchor=(1.0, 0.5))
    pl.title('Hierarchical clustering')
    pl.tight_layout()
    pl.show()

    for i in range(len(k_scores)):
        pl.errorbar(x, k_scores[i],
                    xerr=None, yerr=k_errors[i],
                    fmt='-o', capsize=5,
                    label=labs[i])
    pl.xlabel('Number of clusters')
    pl.ylabel('Gini score')
    pl.legend()  # loc='center left', bbox_to_anchor=(1.0, 0.5))
    pl.title('K-means clustering')
    pl.tight_layout()
    pl.show()
