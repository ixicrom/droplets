from processing_tools import *
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage import io
import numpy as np
from sklearn import cluster
from sklearn.decomposition import PCA
from collections import Counter


# variables for testing_________________________________________________________
filePath='/Users/s1101153/Desktop/droplet_stacks/63x/rect_pickles'

n_clust=5
show_dendro=False

col_name='k-cluster'

cluster_dat=h
plot_col_name='Cluster_hier'
plot_title='Hierarchical cluster plot'


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
    elif scale == 'standard'
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
    dat.insert(0,'phip', phip.str.replace('-', '.').astype(int))

    dat = dat.set_index(['sample', 'colour', 'slice', 'phip', 'phir'])
    dat
    return dat

def read_calc_format_wedges(reslice = False, imPath = None, infoFile = None,
hp = False, scale):
    '''
    Inputs:
        reslice=True: will calculate slices from files in imPath using info from infoFile, and save to '/Users/s1101153/Desktop/droplet_stacks/63x/nice_slices.pkl'
        reslice=False: will read in wedge slices from '/Users/s1101153/Desktop/droplet_stacks/63x/nice_slices.pkl'

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

        scale='minmax': uses MinMaxScaler on all value columns
        scale='standard': uses StandardScaler on all value columns

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
        sFile = input('Input filepath for saving: ')
        slices = slice_folder(imPath, infoFile, save=True, saveFile='/Users/s1101153/Desktop/droplet_stacks/63x/nice_slices.pkl')
    else:
        slices = pd.read_pickle('/Users/s1101153/Desktop/droplet_stacks/63x/nice_slices.pkl')


    slices_data = calc_variables(slices, highpass = hp)

    slices_data = slices_data.drop('imArray', axis=1)

    if scale=='minmax':
        scaler = MinMaxScaler()
        vals_scaled = scaler.fit_transform(slices_data)
        slices_data = pd.DataFrame(vals_scaled, index=slices_data.index, columns=slices_data.columns)

    elif scale == 'standard'
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
    return pca, dat_PCA

def tSNE_transform(dat, p):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=p)
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

def cluster(method, dat, n_clust):
    if method == 'h':
        h_count = hier(dat, n_clust, False)

    elif method == 'k'
        k_count = kmeans(dat, n_clust)
    else:
        print('Invalid method input. Use \'h\' for hierarchical clustering or \'k\' for k-means clustering.')


# functions for visualisation
def gini_score_(method, dat, start_num, stop_num):
    score = list()
    for i in range(start_num, stop_num):
        count = cluster(method, dat, i)
        score.append(gini_score(count))
    return score

def tSNE_plot():



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
