from slice_tools import *
from sklearn import cluster


def main():
    filePath='/Users/s1101153/Dropbox/Emily/'
    # datFile=filePath+'z-stack_info.csv'
    # slice_all(datFile, filePath, save=True)

    dat=read_files(filePath)
    dat = dat.dropna()
    # dat=dat.transpose()
    dat_means=dat.groupby([('T6M_29_1_slice5.pkl','r')]).mean()
    dat_means
    idx=pd.IndexSlice
    dat_mean_green=dat_means.loc[:,idx[:,'val_green']]
    dat_mean_green=dat_mean_green.droplevel(1,axis=1)
    dat_mean_green.index.name='r'
    dat_mean_green
    dat_arr=dat_mean_green.transpose().to_numpy()
    dat_arr.shape

    np.random.seed(1234)
    kmeans=cluster.KMeans(n_clusters=9).fit(dat_arr)

    pred = kmeans.predict(dat_mean_green.transpose())
    pred
    dat_final_g = dat_mean_green.transpose()
    dat_final_g
    dat_final_g['Cluster'] = pd.Series(pred, index = dat_final_g.index)
    dat_print_g=dat_final_g['Cluster'].sort_index()
    # labs=kmeans.labels_
    # labs
    # len(labs)
    # np.savetxt('test_2020-04-14.txt',np.asarray(labs))
    dat_print_g.to_csv('test_2020-04-14_9clusters.csv', header=False)

main()
