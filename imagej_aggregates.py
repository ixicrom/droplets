import pandas as pd
import os
import matplotlib.pyplot as pl
import seaborn as sb

# %% setup
minFiles = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/aggregates_minority/'
majFiles = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/aggregates_majority/'

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

min_col = {'phip1_phir25_2.csv': 'red',
           'phip1_phir25.csv': 'red',
           'phip1_phir40.csv': 'red',
           'phip1_phir60_2.csv': 'green',
           'phip1_phir60.csv': 'green',
           'phip1_phir75_2.csv': 'green',
           'phip1_phir75.csv': 'green',
           'phip0-5_phir10_2.csv': 'red',
           'phip0-5_phir10_3.csv': 'red',
           'phip0-5_phir10_4.csv': 'red',
           'phip0-5_phir10.csv': 'red',
           'phip0-5_phir25.csv': 'red',
           'phip0-5_phir40.csv': 'red',
           'phip0-5_phir60.csv': 'green',
           'phip0-5_phir75.csv': 'green'}

maj_col = {'phip1_phir25_2.csv': 'green',
           'phip1_phir25.csv': 'green',
           'phip1_phir40.csv': 'green',
           'phip1_phir60_2.csv': 'red',
           'phip1_phir60.csv': 'red',
           'phip1_phir75_2.csv': 'red',
           'phip1_phir75.csv': 'red',
           'phip0-5_phir10_2.csv': 'green',
           'phip0-5_phir10_3.csv': 'green',
           'phip0-5_phir10_4.csv': 'green',
           'phip0-5_phir10.csv': 'green',
           'phip0-5_phir25.csv': 'green',
           'phip0-5_phir40.csv': 'green',
           'phip0-5_phir60.csv': 'red',
           'phip0-5_phir75.csv': 'red'}

# %% read in data from files
agg_frac = []
droplet_cat = []
colour = []
for file in os.listdir(minFiles):
    if file.endswith('.csv'):
        f = os.path.join(minFiles, file)
        dat = pd.read_csv(f, index_col=0)
        agg_dat = dat['Area'][dat['Area'] > 1]
        agg_area = agg_dat.sum()
        tot_area = dat['Area'].sum()
        agg_frac.append(agg_area/tot_area)
        droplet_cat.append(categories[file])
        colour.append(min_col[file])

agg_frac_maj = []
maj_drop_cat = []
colour_maj = []
for file in os.listdir(majFiles):
    if file.endswith('.csv'):
        f = os.path.join(majFiles, file)
        dat = pd.read_csv(f, index_col=0)
        agg_dat = dat['Area'][dat['Area'] > 1]
        agg_area = agg_dat.sum()
        tot_area = dat['Area'].sum()
        agg_frac_maj.append(agg_area/tot_area)
        maj_drop_cat.append(categories[file])
        colour_maj.append(maj_col[file])

min_dat = pd.DataFrame([droplet_cat, agg_frac, colour]).transpose()
min_dat.columns = ['Droplet shape', 'Aggregate area fraction', 'Particle']
min_dat['Channel'] = ['Minority']*min_dat.shape[0]

maj_dat = pd.DataFrame([maj_drop_cat, agg_frac_maj, colour_maj]).transpose()
maj_dat.columns = ['Droplet shape', 'Aggregate area fraction', 'Particle']
maj_dat['Channel'] = ['Majority']*maj_dat.shape[0]

plot_dat = pd.concat([maj_dat, min_dat]).reset_index()
plot_dat['Aggregate area fraction'] = plot_dat['Aggregate area fraction'].astype('float64')
plot_dat

# %% min/maj split plot
g = sb.violinplot(data=plot_dat,
                  x='Droplet shape',
                  y='Aggregate area fraction',
                  hue='Channel',
                  split=True,
                  inner='quartile',
                  linewidth=1,
                  palette='pastel',
                  legend=False,
                  scale='area')
sb.swarmplot(data=plot_dat,
             x='Droplet shape',
             y='Aggregate area fraction',
             hue='Channel')
pl.legend(loc='lower left')
pl.title('Aggregate area fraction')
pl.show()

# %% min/maj split plot with outlier removed (not sure this is valid though)
g = sb.violinplot(data=plot_dat.drop(24),
                  x='Droplet shape',
                  y='Aggregate area fraction',
                  hue='Channel',
                  split=True,
                  inner='quartile',
                  linewidth=1,
                  palette='pastel',
                  legend=False)
sb.swarmplot(data=plot_dat.drop(24),
             x='Droplet shape',
             y='Aggregate area fraction',
             hue='Channel')
pl.legend(loc='lower right')
pl.title('Aggregate area fraction (outliers removed)')
pl.show()

# %% red/green split plot
g = sb.violinplot(data=plot_dat,
                  x='Droplet shape',
                  y='Aggregate area fraction',
                  hue='Particle',
                  split=True,
                  inner='quartile',
                  linewidth=1,
                  palette='pastel',
                  legend=False)
sb.swarmplot(data=plot_dat,
             x='Droplet shape',
             y='Aggregate area fraction',
             hue='Particle')
pl.legend(loc='lower left')
pl.title('Aggregate area fraction by particle type')
pl.show()

# %% min/maj side-by-side plots
g = sb.violinplot(data=plot_dat,
                  x='Channel',
                  y='Aggregate area fraction',
                  hue='Droplet shape',
                  inner='quartile',
                  linewidth=1,
                  palette='pastel',
                  legend=False,
                  scale='area')
pl.legend(loc='lower left')
pl.show()

# %% red/green side-by-side plots
g = sb.violinplot(data=plot_dat,
                  x='Particle',
                  y='Aggregate area fraction',
                  hue='Droplet shape',
                  inner='quartile',
                  linewidth=1,
                  palette='pastel',
                  legend=False,
                  scale='area')
pl.legend(loc='lower left')
pl.show()
