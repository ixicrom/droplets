from slice_tools import *
from sklearn import cluster


def main():
    filePath='/Users/s1101153/Dropbox/Emily/'
    # datFile=filePath+'z-stack_info.csv'
    # slice_all(datFile, filePath, save=True)

    dat=read_files(filePath)
    dat = dat.dropna()
    dat=dat.transpose()

    dat=dat.to_numpy()
    dat.shape

    # type(dat)
    # type(dat[0])
    big_array=np.full((108,4,24400), np.nan)
    for i in range(108):
        j=i*4
        little_array=dat[j:j+4, :]
        big_array[i,]=little_array

    kmeans=cluster.KMeans(n_clusters=8).fit(big_array[:,3,:])
    labs=kmeans.labels_
    len(labs)
    np.savetxt('test.txt',np.asarray(labs))

main()
