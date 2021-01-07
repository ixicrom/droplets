import pandas as pd
import os
import matplotlib.pyplot as pl
import seaborn as sb

# %% setup
minFiles = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/edge_shape_minority/'
majFiles = '/Users/s1101153/OneDrive - University of Edinburgh/Files/OCP_working/droplet_stacks/edge_shape_majority/'

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

# %% read in data from both channels
drop_circ = []
area_perim = []
round = []
droplet_cat = []
colour = []
for file in os.listdir(minFiles):
    if file.endswith('.csv'):
        f = os.path.join(minFiles, file)
        dat = pd.read_csv(f, index_col=0)
        a_p = (dat['Area']/dat['Perim.']).values[0]
        drop_circ.append(dat['Circ.'].values[0])
        area_perim.append(a_p)
        r = (dat['Circ.']/dat['Round']).values[0]
        round.append(r)
        droplet_cat.append(categories[file])
        colour.append(min_col[file])

drop_circ_maj = []
area_perim_maj = []
agg_frac_maj = []
round_maj = []
drop_cat_maj = []
colour_maj = []
for file in os.listdir(majFiles):
    if file.endswith('.csv'):
        f = os.path.join(majFiles, file)
        dat = pd.read_csv(f, index_col=0)
        a_p = (dat['Area']/dat['Perim.']).values[0]
        drop_circ_maj.append(dat['Circ.'].values[0])
        area_perim_maj.append(a_p)
        r = (dat['Circ.']/dat['Round']).values[0]
        round_maj.append(r)
        drop_cat_maj.append(categories[file])
        colour_maj.append(maj_col[file])

min_dat = pd.DataFrame([droplet_cat, drop_circ, area_perim, round, colour]).transpose()
min_dat.columns = ['Droplet shape', 'Droplet circularity', 'Droplet area per perimeter', 'Droplet circularity/roundness', 'Particle']
min_dat['Channel'] = ['Minority']*min_dat.shape[0]
min_dat['Droplet circularity'] = min_dat['Droplet circularity'].astype('float64')
min_dat['Droplet area per perimeter'] = min_dat['Droplet area per perimeter'].astype('float64')
min_dat['Droplet circularity/roundness'] = min_dat['Droplet circularity/roundness'].astype('float64')

maj_dat = pd.DataFrame([drop_cat_maj, drop_circ_maj, area_perim_maj, round_maj, colour_maj]).transpose()
maj_dat.columns = ['Droplet shape', 'Droplet circularity', 'Droplet area per perimeter', 'Droplet circularity/roundness', 'Particle']
maj_dat['Channel'] = ['Majority']*maj_dat.shape[0]
maj_dat['Droplet circularity'] = maj_dat['Droplet circularity'].astype('float64')
maj_dat['Droplet area per perimeter'] = maj_dat['Droplet area per perimeter'].astype('float64')
maj_dat['Droplet circularity/roundness'] = maj_dat['Droplet circularity/roundness'].astype('float64')


plot_dat = pd.concat([maj_dat, min_dat]).reset_index()
# plot_dat['Droplet circularity'] = plot_dat['Droplet circularity'].astype('float64')
# plot_dat['Droplet area per perimeter'] = plot_dat['Droplet area per perimeter'].astype('float64')

# %% min/maj split plot circularity
g = sb.violinplot(data=plot_dat,
                  x='Droplet shape',
                  y='Droplet circularity',
                  hue='Channel',
                  split=True,
                  inner='quartile',
                  linewidth=1,
                  palette='pastel',
                  legend=False,
                  scale='area')
sb.swarmplot(data=plot_dat,
             x='Droplet shape',
             y='Droplet circularity',
             hue='Channel')
pl.legend(loc='lower right')
pl.title('Droplet circularity')
pl.show()

# %% min/maj split plot area/perimeter
g = sb.violinplot(data=plot_dat,
                  x='Droplet shape',
                  y='Droplet area per perimeter',
                  hue='Channel',
                  split=True,
                  inner='quartile',
                  linewidth=1,
                  palette='pastel',
                  legend=False)
sb.swarmplot(data=plot_dat,
             x='Droplet shape',
             y='Droplet area per perimeter',
             hue='Channel')
pl.legend(loc='upper center')
pl.title('Droplet area per perimeter')
pl.show()

# %% red/green split plot circularity
g = sb.violinplot(data=plot_dat,
                  x='Droplet shape',
                  y='Droplet circularity',
                  hue='Particle',
                  split=True,
                  inner='quartile',
                  linewidth=1,
                  palette='pastel',
                  legend=False,
                  scale='area')
sb.swarmplot(data=plot_dat,
             x='Droplet shape',
             y='Droplet circularity',
             hue='Particle')
pl.legend(loc='lower right')
pl.title('Droplet circularity (by particle type)')
pl.show()

# %% red/green split plot area/perimeter
g = sb.violinplot(data=plot_dat,
                  x='Droplet shape',
                  y='Droplet area per perimeter',
                  hue='Particle',
                  split=True,
                  inner='quartile',
                  linewidth=1,
                  palette='pastel',
                  legend=False)
sb.swarmplot(data=plot_dat,
             x='Droplet shape',
             y='Droplet area per perimeter',
             hue='Particle')
pl.legend(loc='upper center')
pl.title('Droplet area per perimeter (by particle type)')
pl.show()

# %% area/perimeter plot all data
# g = sb.violinplot(data=plot_dat,
#                   x='Droplet shape',
#                   y='Droplet area per perimeter',
#                   inner='quartile',
#                   palette='pastel')
sb.swarmplot(data=plot_dat,
             x='Droplet shape',
             y='Droplet area per perimeter',
             size=10)
pl.title('All channels droplet area per perimeter')
pl.savefig('/Users/s1101153/Desktop/area_per_perim_all.png')
pl.show()

# %% circularity plot all data
# g = sb.violinplot(data=plot_dat,
#                   x='Droplet shape',
#                   y='Droplet circularity',
#                   inner='quartile',
#                   palette='pastel')
sb.swarmplot(data=plot_dat,
             x='Droplet shape',
             y='Droplet circularity',
             size=10)
pl.title('All channels droplet circularity')
pl.savefig('/Users/s1101153/Desktop/circ_all.png')
pl.show()

# %% area/perimeter plot majority channel
# g = sb.violinplot(data=maj_dat,
#                   x='Droplet shape',
#                   y='Droplet area per perimeter',
#                   inner='quartile',
#                   palette='pastel')
sb.swarmplot(data=maj_dat,
             x='Droplet shape',
             y='Droplet area per perimeter',
             size=10)
pl.title('Majority channel droplet area per perimeter')
pl.savefig('/Users/s1101153/Desktop/area_per_perim_maj.png')
pl.show()

# %% circularity plot majority channel
# g = sb.violinplot(data=maj_dat,
#                   x='Droplet shape',
#                   y='Droplet circularity',
#                   inner='quartile',
#                   palette='pastel')
sb.swarmplot(data=maj_dat,
             x='Droplet shape',
             y='Droplet circularity',
             size=10)
pl.title('Majority channel droplet circularity')
pl.savefig('/Users/s1101153/Desktop/circ_maj.png')
pl.show()

# %% circularity/roundness plot all data
# g = sb.violinplot(data=plot_dat,
#                   x='Droplet shape',
#                   y='Droplet circularity/roundness',
#                   inner='quartile',
#                   palette='pastel')
sb.swarmplot(data=plot_dat,
             x='Droplet shape',
             y='Droplet circularity/roundness',
             size=10)
pl.title('All channels droplet circularity per roundness')
pl.savefig('/Users/s1101153/Desktop/circ_per_round_all.png')
pl.show()

# %% circularity/roundness plot majority channel
# g = sb.violinplot(data=maj_dat,
#                   x='Droplet shape',
#                   y='Droplet circularity/roundness',
#                   inner='quartile',
#                   palette='pastel')
sb.swarmplot(data=maj_dat,
             x='Droplet shape',
             y='Droplet circularity/roundness',
             size=10)
pl.title('Majority channel droplet circularity per roundness')
pl.savefig('/Users/s1101153/Desktop/circ_per_round_maj.png')
pl.show()

# %% circularity/roundness plot minority channel
# g = sb.violinplot(data=min_dat,
#                   x='Droplet shape',
#                   y='Droplet circularity/roundness',
#                   inner='quartile',
#                   palette='pastel')
sb.swarmplot(data=min_dat,
             x='Droplet shape',
             y='Droplet circularity/roundness',
             size=10)
pl.title('Minority channel droplet circularity per roundness')
pl.savefig('/Users/s1101153/Desktop/circ_per_round_min.png')
pl.show()


# %% circularity/roundness boxplot majority channel
g = sb.boxplot(data=maj_dat,
               x='Droplet shape',
               y='Droplet circularity/roundness',
               palette='pastel')
sb.swarmplot(data=maj_dat,
             x='Droplet shape',
             y='Droplet circularity/roundness',
             size=10)
pl.title('Majority channel droplet circularity per roundness')
pl.savefig('/Users/s1101153/Desktop/circ_per_round_box.png')
pl.show()
