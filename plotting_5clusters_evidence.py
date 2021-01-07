from full_analysis_tools import read_files, format_rectangles, read_calc_format_wedges, clust, optimalK, gini_score, PCA_transform
import matplotlib.pyplot as pl
import os
import numpy as np
import matplotlib
matplotlib.style.core.reload_library()

graph_folder = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP/Graphs/'

# %% rectangular/theta-averaged data
filePath = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/rect_pickles'
imagePath = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/final_images/ims_to_read/'

dat = read_files(filePath)

r_dat_theta = format_rectangles(dat, scale='standard', theta_av=True)

r_dat_rect = format_rectangles(dat, scale='standard', theta_av=False)


# %% calculate wedge data
wedge_path = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/63x/'


info_file_A = os.path.join(wedge_path, 'stack_info_2020-08-28_A.csv')
save_file_A = os.path.join(wedge_path, 'wedges_A')+'.pkl'


info_file_B = os.path.join(wedge_path, 'stack_info_2020-08-28_B.csv')
save_file_B = os.path.join(wedge_path, 'wedges_B')+'.pkl'

if input('Calculate wedges from scratch? y/n: ') == 'y':
    wedges_A = read_calc_format_wedges(scale='minmax',
                                       fileName=save_file_A,
                                       reslice=True,
                                       imPath=imagePath,
                                       infoFile=info_file_A,
                                       hp=False)
    wedges_B = read_calc_format_wedges(scale='minmax',
                                       fileName=save_file_B,
                                       reslice=True,
                                       imPath=imagePath,
                                       infoFile=info_file_B,
                                       hp=False)

else:
    wedges_A = read_calc_format_wedges(scale='minmax',
                                       fileName=save_file_A,
                                       reslice=False,
                                       hp=False)

    wedges_B = read_calc_format_wedges(scale='minmax',
                                       fileName=save_file_B,
                                       reslice=False,
                                       hp=False)

r_dat_oneA = r_dat_rect[r_dat_rect.index.isin(wedges_A.index)]
r_dat_oneB = r_dat_rect[r_dat_rect.index.isin(wedges_B.index)]

r_dat_oneA = PCA_transform(r_dat_oneA, 0.99)[1]
r_dat_oneB = PCA_transform(r_dat_oneB, 0.99)[1]
r_dat_pca = PCA_transform(r_dat_rect, 0.99)[1]


# %% gini score plots
if input('Make gini score plot? y/n: ') == 'y':
    score_rect = []
    score_theta = []
    score_pca = []
    score_oneA = []
    score_oneB = []

    for i in range(3, 23):
        count_rect = clust('h', r_dat_rect, i, col_name='Hier_cluster')[1]
        count_theta = clust('h', r_dat_theta, i, col_name='Hier_cluster')[1]
        count_pca = clust('h', r_dat_pca, i, col_name='Hier_cluster')[1]
        count_oneA = clust('h', r_dat_oneA, i, col_name='Hier_cluster')[1]
        count_oneB = clust('h', r_dat_oneB, i, col_name='Hier_cluster')[1]
        score_rect.append(np.mean(gini_score(count_rect)))
        score_theta.append(np.mean(gini_score(count_theta)))
        score_pca.append(np.mean(gini_score(count_pca)))
        score_oneA.append(np.mean(gini_score(count_oneA)))
        score_oneB.append(np.mean(gini_score(count_oneB)))

    x = [* range(3, 23)]
    pl.plot(x, score_rect, '-o', label='Rectangular')
    pl.plot(x, score_theta, '-o', label='Theta-averaged')
    pl.plot(x, score_pca, '-o', label='PCA 99\%')
    pl.plot(x, score_oneA, '-o', label='oneA PCA 99\%')
    pl.plot(x, score_oneB, '-o', label='oneB PCA 99\%')
    pl.legend(loc='lower right')
    pl.xlabel('Number of clusters')
    pl.ylabel('Gini score')
    pl.title('Gini score for hierchical clustering', loc='center', wrap=True)
    pl.vlines(5, 0, 0.35, linestyles='dashed', colors='k')
    pl.savefig(graph_folder+'gini_method_overview.png')
    pl.show()

# %% gap statistic plots
if input('Make gap statistic plot? y/n: ') == 'y':
    gap_rect = optimalK(r_dat_rect, maxClusters=23)
    gap_theta = optimalK(r_dat_theta, maxClusters=23)
    gap_pca = optimalK(r_dat_pca, maxClusters=23)
    gap_oneA = optimalK(r_dat_oneA, maxClusters=23)
    gap_oneB = optimalK(r_dat_oneB, maxClusters=23)

    gap_rect[1]['gap'][0]

    pl.plot(gap_rect[1]['clusterCount'],
            gap_rect[1]['gap']-gap_rect[1]['gap'][0],
            '-o',
            label='Rectangular')
    pl.plot(gap_theta[1]['clusterCount'],
            gap_theta[1]['gap']-gap_theta[1]['gap'][0],
            '-o',
            label='Theta-averaged')
    pl.plot(gap_pca[1]['clusterCount'],
            gap_pca[1]['gap']-gap_pca[1]['gap'][0],
            '-o',
            label='PCA 99\%')
    pl.plot(gap_oneA[1]['clusterCount'],
            gap_oneA[1]['gap']-gap_oneA[1]['gap'][0],
            '-o',
            label='oneA PCA 99\%')
    pl.plot(gap_oneB[1]['clusterCount'],
            gap_oneB[1]['gap']-gap_oneB[1]['gap'][0],
            '-o',
            label='oneB PCA 99\%')
    pl.xlabel('Number of clusters')
    pl.ylabel('Gap statistic (offset to start at 0)')
    pl.title('Gap statistic for k-means clustering', wrap=True)
    pl.legend(loc='upper left')
    pl.vlines(5, 0, 1.75, linestyles='dashed', colors='k')
    pl.savefig(graph_folder+'gap_method_overview.png', bbox_inches='tight')
    pl.tight_layout()
    pl.show()
