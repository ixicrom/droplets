import matplotlib.pyplot as plt
import pandas as pd
from data_tools import *

dat = pd.read_csv('/Users/s1101153/Dropbox/Emily/Results/2020-05-15_ACF_clusters_1d-av.csv', index_col=0)
dat = dat.drop(['ACF_index', 'Value'], axis='columns').drop_duplicates().reset_index(drop=True)

cluster_count = count_clusters(dat, 'Sample', 'Colour', counter='Cluster')

cluster_count.transpose().plot(kind='bar', stacked = True)
