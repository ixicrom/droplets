from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pl
from sklearn import manifold
from processing_tools import * # also imports all from data_tools and slice_tools


filePath='/Users/s1101153/Desktop/droplet_stacks/63x/rect_pickles'
dat=read_files(filePath)
idx = pd.IndexSlice
colms=['val_green', 'val_red']
vals = dat.loc[:,idx[:,colms]].transpose()
vals
vals.shape

# vals.to_numpy()


scaler = StandardScaler()
scaled_vals=scaler.fit_transform(vals)
scaled_vals.shape

perp_vals = [5,10,30,50,80,100]

i=1
for p in perp_vals:
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=p)
    dat_tsne = tsne.fit_transform(scaled_vals)
    pl.subplot(2,3,i)
    pl.scatter(dat_tsne[:,0], dat_tsne[:,1])
    pl.title('Perplexity = '+str(p))
    i=i+1
pl.show()


dat_tsne.shape
