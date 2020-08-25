from processing_tools import *
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage import io
import numpy as np
from sklearn import cluster
from sklearn.decomposition import PCA
from collections import Counter
from sklearn import manifold

# variables for testing_________________________________________________________
# filePath='/Users/s1101153/Desktop/droplet_stacks/63x/rect_pickles'
#
# n_clust=5
# show_dendro=False
#
# col_name='k-cluster'
#
# cluster_dat=h
# plot_col_name='Cluster_hier'
# plot_title='Hierarchical cluster plot'


# functions for getting data in desired format__________________________________
def read_format_rectangles(filePath, scale, theta_av=True):
    '''
    Reads in pickles from filePath and reformats dataframe ready for clustering.
    Initial format:
        27300 rows × 768 columns
        Columns:
            vars = 0 to r*theta=27300
            slices = pickle file names
                r = 0 to 272
                theta = 0 to 100
                val_green = green channel values for each r/theta combo
                val_red = red channel values for each r/theta combo

    Options:
        scale='minmax': uses MinMaxScaler on all value columns
        scale='standard': uses StandardScaler on all value columns

        theta_av = True: averages over theta to give 1D array using theta_average function in data_tools.py
        theta_av = False: does nothing

    Output:
        theta_av=True:
            384 rows x 273 columns
            MultiIndex:
                sample = sample ID eg phip0-5_phir10_2
                colour = green or red
                slice = 0 to 11
                phip
                phir
            Columns:
                0 to 272 (each r value)
        theta_av=False:
            384 rows x 27300 columns
            MultiIndex:
                sample = sample ID eg phip0-5_phir10_2
                colour = green or red
                slice = 0 to 11
                phip
                phir
            Columns:
                0 to 27299 (each r x theta combo)

    '''

    dat = read_files(filePath)
    idx = pd.IndexSlice

    if theta_av:
        dat = theta_average(dat).transpose()
    else:
        colms = ['val_green', 'val_red']
        vals = dat.loc[:,idx[:,colms]].transpose()
        dat = pd.DataFrame(vals, index=vals.index)


    if scale=='minmax':
        scaler = MinMaxScaler()
        vals_scaled = scaler.fit_transform(dat)
        dat = pd.DataFrame(vals_scaled, index=dat.index)
    elif scale == 'standard':
        scaler = StandardScaler()
        vals_scaled = scaler.fit_transform(dat)
        dat = pd.DataFrame(vals_scaled, index=dat.index)
    else:
        print('No scaling applied. If scaling desired, set scale to \'minmax\' or \'standard\'.')

    dat = dat.reset_index()

    samples = dat['slices'].str.slice(4,21).str.rstrip('_stack')
    dat.insert(0, 'sample', samples)


    slice_nums = dat['slices'].str.rstrip('.pkl').str[-2:].str.lstrip('e').astype(int)
    dat['slices'] = slice_nums

    dat = dat.rename(columns={'slices':'slice'})

    colours = dat['vars'].str[4:]
    dat.insert(1,'colour', colours)

    dat = dat.drop('vars', axis=1)

    phip=samples.str[0:7].str.strip('_phi')
    phir=samples.str[10:15].str.lstrip('phir').str.rstrip('_2')

    dat.insert(0,'phir', phir.astype(int))
    dat.insert(0,'phip', phip.str.replace('-', '.').astype(float))

    dat = dat.set_index(['sample', 'colour', 'slice', 'phip', 'phir'])
    dat
    return dat

def read_calc_format_wedges(scale, fileName, reslice, imPath = None, infoFile = None, hp = False):
    '''
    Inputs:
        scale='minmax': uses MinMaxScaler on all value columns
        scale='standard': uses StandardScaler on all value columns

        fileName: file to either read from or save to

        reslice=True: will calculate slices from files in imPath using info from infoFile, and save to fileName
        reslice=False: will read in wedge slices from fileName

        imPath: folder for reslicing. Should contain tif images to be sliced, each with 2 channels (green and red). Shape should be (n, n, 2).

        infoFile: csv file with info for reslicing. Contains columns:
            1. image file names (from imPath) to be used
            2. size of image in um
            3. size of image in pixels
            4. centre of droplet x coordinate in um
            5. centre of droplet y coordinate in um
            6. location (0 or 1) of green channel in image
            7. location (0 or 1) of red channel in image
            8. smallest radius of the droplet (not used, may be later)
            9. largest radius of the droplet (not used)

        hp: toggle for whether to apply a highpass filter to the images before thresholding


    Output:
        384 rows x 6 columns
        MultiIndex:
            sample = sample ID eg phip0-5_phir10_2
            colour = green or red
            slice = 0 to 11
        Columns:
            'av_connected_area', 'stdev_connected_area', 'av_separation',        'stdev_separation', 'area_sum', 'av_circularity'

    '''

    if reslice:
        print('Re-calculating slices...')
        # sFile = input('Input filepath for saving: ')
        slices = slice_folder(imPath, infoFile, save=True, saveFile=fileName)
    else:
        slices = pd.read_pickle(fileName)


    slices_data = calc_variables(slices, highpass = hp)

    slices_data = slices_data.reset_index()
    samples = slices_data['sample']#.str.slice(4,21).str.rstrip('_stack')
    phip=samples.str[0:7].str.strip('_phi')
    phir=samples.str[10:15].str.lstrip('phir').str.rstrip('_2')

    slices_data.insert(0,'phir', phir.astype(int))
    slices_data.insert(0,'phip', phip.str.replace('-', '.').astype(float))

    slices_data = slices_data.set_index(['sample', 'colour', 'slice', 'phip', 'phir'])

    slices_data = slices_data.drop('imArray', axis=1).dropna()

    if scale=='minmax':
        scaler = MinMaxScaler()
        vals_scaled = scaler.fit_transform(slices_data)
        slices_data = pd.DataFrame(vals_scaled, index=slices_data.index, columns=slices_data.columns)

    elif scale == 'standard':
        scaler = StandardScaler()
        vals_scaled = scaler.fit_transform(slices_data)
        slices_data = pd.DataFrame(vals_scaled, index=slices_data.index, columns=slices_data.columns)
    else:
        print('No scaling applied. If scaling desired, set scale to \'minmax\' or \'standard\'.')

    return slices_data

# functions for transforming the data into lower dimensions_____________________
def PCA_transform(dat, var):
    np.random.seed(1234)
    pca = PCA(var).fit(dat)
    dat_PCA = pca.transform(dat)
    dat_PCA = pd.DataFrame(dat_PCA, index=dat.index)
    return pca, dat_PCA

def tSNE_transform(dat, p, r_state=0):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=r_state, perplexity=p)
    dat_tsne = tsne.fit_transform(dat)
    return dat_tsne


# functions for clustering that all return the output of count_clusters_________
def hier(dat, n_clust, show_dendro=False):
    np.random.seed(1234)
    h = h_cluster(dat, n_clust, showPlot=show_dendro)
    h.index = dat.index
    h_count = count_clusters(h.reset_index(), counter='Cluster_hier',
                            grouper1='sample', grouper2='colour')
    return h, h_count


def kmeans(dat, n_clust, col_name):
    np.random.seed(1234)
    km = cluster.KMeans(n_clusters = n_clust)
    km_fit = km.fit(dat)
    labs = pd.DataFrame(km_fit.labels_, index=dat.index, columns=[col_name])
    k_count = count_clusters(labs.reset_index(), counter=col_name,
                            grouper1='sample', grouper2 = 'colour')
    return labs, k_count

def clust(method, dat, n_clust, col_name=""):
    if method == 'h':
        count = hier(dat, n_clust, False)

    elif method == 'k':
        count = kmeans(dat, n_clust, col_name)
    else:
        print('Invalid method input. Use \'h\' for hierarchical clustering or \'k\' for k-means clustering.')
    return count


# functions for visualisation
def gini_score_range(method, dat, start_num, stop_num):
    score = list()
    for i in range(start_num, stop_num):
        count = clust(method, dat, i, col_name='cluster')[1]
        score.append(np.mean(gini_score(count)))
    return score

def tSNE_plot(dat_tsne, col_dat, plot_title, **pl_kwargs):
    '''
    Plots col_dat on coordinates specified in dat_tsne.

    Inputs:
        dat_tsne: n x 2 DataFrame
        col_dat: n x 1 series or index, in same order as dat_tsne
        plot_title: string
        **pl_kwargs: additional inputs for plotting with pl.scatter. Suggest setting alpha=0.7.

    Outputs:
        Displays a scatter plot of each coordinate in dat_tsne, with colour

    '''
    col_unique = col_dat.unique().values
    n_colours = len(col_unique)
    if n_colours>10:
        cmap = pl.get_cmap('tab20')
        plot_colours = cmap(np.linspace(start=0, stop=1, num=n_colours))
    elif n_colours>5:
        cmap = pl.get_cmap('tab10')
        plot_colours = cmap(np.linspace(start=0, stop=1, num=n_colours))
    else:
        cmap = pl.get_cmap('tab10')
        plot_colours = cmap(np.linspace(start=0, stop=0.5, num=n_colours))

    clr = {col_unique[i]: plot_colours[i] for i in range(n_colours)}

    for point in range(dat_tsne.shape[0]):
        pl.scatter(dat_tsne[point,0], dat_tsne[point,1],
                    color = clr[col_dat[point]],
                    label = col_dat[point],
                    **pl_kwargs)
    handles, labels = pl.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    pl.title(plot_title)
    pl.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.0,0.5))
    pl.show()

def phi_plot(cluster_dat, plot_col_name, plot_title):
    cluster_dat
    cmap = pl.get_cmap('tab10')

    clusters=np.unique(cluster_dat[plot_col_name].values)
    clusters

    plot_colours = cmap(np.linspace(start=0, stop=0.5, num=len(clusters)))
    clr = {i: plot_colours[i] for i in clusters}

    phip=cluster_dat.index.get_level_values(3).values
    phir = cluster_dat.index.get_level_values(4).values
    points = zip(phip, phir, cluster_dat[plot_col_name])
    c=Counter(points)

    for point in c.keys():
        pl.scatter(point[0], point[1],
            s=(c[point]**3),
            edgecolors=clr[point[2]],
            facecolors='none',
            label=point[2])
    handles, labels = pl.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    pl.title(plot_title+'\n(size indicates number of points in that position)')
    pl.xlabel('phi_p')
    pl.ylabel('phi_r')
    leg = pl.legend(by_label.values(), by_label.keys(), loc='center left',
                    bbox_to_anchor=(1.0,0.5), title='Cluster')
    for i in clusters:
        leg.legendHandles[i]._sizes=[30]
    pl.show()


def bar_stack(count_dat):
    count_dat.transpose().plot(kind='bar', stacked=True)
    pl.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
    pl.show()
