from slice_tools import *
from sklearn import cluster


def main():
    filePath='/Users/s1101153/Dropbox/Emily/'
    # datFile=filePath+'z-stack_info.csv'
    # slice_all(datFile, filePath, save=True)

    dat=read_files(filePath)
    dat = dat.dropna()
    dat=dat.transpose()
    dat_means=dat.groupby([('T6M_29_1_slice5.pkl','r')]).mean()
    dat_means
    idx=pd.IndexSlice
    dat_mean_green=dat_means.loc[:,idx[:,'val_green']]
    dat_mean_green=dat_mean_green.droplevel(1,axis=1)
    dat_mean_green.index.name='r'
    dat_mean_green
    dat_arr=dat_mean_green.transpose().to_numpy()
    dat_arr.shape

    kmeans=cluster.KMeans().fit(dat_arr)

    labs=kmeans.labels_
    len(labs)
    np.savetxt('test_2020-04-09.txt',np.asarray(labs))

main()
