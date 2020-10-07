import pandas as pd
import os
import matplotlib.pyplot as pl
import seaborn as sb

minFiles = '/Users/s1101153/Desktop/droplet_stacks/aggregates_minority/'
majFiles = '/Users/s1101153/Desktop/droplet_stacks/aggregates_majority/'

categories = {'phip1_phir25_2.csv': 'round',
              'phip1_phir25.csv': 'distorted',
              'phip1_phir40.csv': 'round',
              'phip1_phir60_2.csv': 'round',
              'phip1_phir60.csv': 'round',
              'phip1_phir75_2.csv': 'round',
              'phip1_phir75.csv': 'round-ish',
              'phip0-5_phir10_2.csv': 'distorted',
              'phip0-5_phir10_3.csv': 'distorted',
              'phip0-5_phir10_4.csv': 'round-ish',
              'phip0-5_phir10.csv': 'distorted',
              'phip0-5_phir25.csv': 'round-ish',
              'phip0-5_phir40.csv': 'round',
              'phip0-5_phir60.csv': 'round-ish',
              'phip0-5_phir75.csv': 'round'}

agg_frac = []
droplet_cat = []

for file in os.listdir(minFiles):
    if file.endswith('.csv'):
        f = os.path.join(minFiles, file)
        dat = pd.read_csv(f, index_col=0)
        agg_dat = dat['Area'][dat['Area'] > 1]
        agg_area = agg_dat.sum()
        tot_area = dat['Area'].sum()
        agg_frac.append(agg_area/tot_area)
        droplet_cat.append(categories[file])
        # print(file)

agg_frac_maj = []


for file in os.listdir(majFiles):
    if file.endswith('.csv'):
        f = os.path.join(majFiles, file)
        dat = pd.read_csv(f, index_col=0)
        agg_dat = dat['Area'][dat['Area'] > 1]
        agg_area = agg_dat.sum()
        tot_area = dat['Area'].sum()
        agg_frac_maj.append(agg_area/tot_area)

plot_dat = pd.DataFrame([droplet_cat, agg_frac, agg_frac_maj])
plot_dat = plot_dat.transpose()
plot_dat.columns = ['Droplet shape', 'agg_frac_min', 'agg_frac_maj']
plot_dat

g = sb.boxplot(data=plot_dat,
               x='Droplet shape',
               y='agg_frac_min',
               palette='pastel').set_title('Aggregate fraction')
sb.swarmplot(data=plot_dat,
             x='Droplet shape',
             y='agg_frac_min',
             color='k')
pl.show()


g_maj = sb.boxplot(data=plot_dat[plot_dat.notnull()],
                   x='Droplet shape',
                   y='agg_frac_maj',
                   palette='pastel').set_title('Aggregate fraction')
sb.swarmplot(data=plot_dat[plot_dat.notnull()],
             x='Droplet shape',
             y='agg_frac_maj',
             color='k')
pl.show()




dat.describe()
dat['Area'].quantile(q=0.75)
dat['Area'].plot(kind='hist')
