import pandas as pd
import os
import matplotlib.pyplot as pl
from datetime import datetime


# %% initialise empty lists
circ_mean05 = []
round_mean05 = []
circ_min05 = []
round_min05 = []
circ_count05 = []
round_count05 = []
phir05 = []

circ_mean1 = []
round_mean1 = []
circ_min1 = []
round_min1 = []
circ_count1 = []
round_count1 = []
phir1 = []


# %%

min_flag = input('Absolute minimum (abs) or averaged minimum 10% (ave): ')
filePath = '/Users/s1101153/Desktop/image_analysis/'

for file in os.listdir(filePath):
    if file.endswith('.csv'):
        f = os.path.join(filePath, file)
        dat = pd.read_csv(f, index_col=0).iloc[:, :11]
        c_dat = dat[['Circ.', 'Round']].dropna()
        info = file[4:-10]
        phip = info.rstrip('phir0124567')[:-1]
        phir = info.lstrip('phip0-51')[1:]

        if min_flag == 'abs':
            circ_min = c_dat.min()[0]
            round_min = c_dat.min()[1]
        elif min_flag == 'ave':
            num_average = int(len(c_dat)/10)
            circ_min = c_dat['Circ.'].sort_values()[:num_average].mean()
            round_min = c_dat['Round'].sort_values()[:num_average].mean()
        else:
            print('No suitable flag provided. Code will crash.')

        if phip.endswith('0-5'):
            circ_mean05.append(c_dat.mean()[0])
            round_mean05.append(c_dat.mean()[1])
            circ_min05.append(circ_min)
            round_min05.append(round_min)
            cc = 1-len(c_dat[c_dat['Circ.'] < 0.9])/len(c_dat['Circ.'])
            circ_count05.append(cc)
            rc = 1-len(c_dat[c_dat['Round'] < 0.9])/len(c_dat['Round'])
            round_count05.append(rc)
            phir05.append(phir.lstrip('phir'))

        elif phip.endswith('1'):
            circ_mean1.append(c_dat.mean()[0])
            round_mean1.append(c_dat.mean()[1])
            circ_min1.append(circ_min)
            round_min1.append(round_min)
            cc = 1-len(c_dat[c_dat['Circ.'] < 0.9])/len(c_dat['Circ.'])
            circ_count1.append(cc)
            rc = 1-len(c_dat[c_dat['Round'] < 0.9])/len(c_dat['Round'])
            round_count1.append(rc)
            phir1.append(phir.lstrip('phir'))


circ_data1 = pd.DataFrame(list(zip(circ_mean1, round_mean1,
                                   circ_min1, round_min1)),
                          columns=['circ_mean1', 'round_mean1',
                                   'circ_min1', 'round_min1'],
                          index=phir1).sort_index()

circ_data05 = pd.DataFrame(list(zip(circ_mean05, round_mean05,
                                    circ_min05, round_min05)),
                           columns=['circ_mean05', 'round_mean05',
                                    'circ_min05', 'round_min05'],
                           index=phir05).sort_index()

circ_data1.plot()
pl.xlabel('phi_r')
pl.ylabel('circularity measure')
if min_flag == 'abs':
    pl.title('phi_p = 1, minimum = single smallest value')
elif min_flag == 'ave':
    pl.title('phi_p = 1, minimum = average of smallest 10%')
time = datetime.now().strftime('%Y-%m-%d_%H%M')
savePath = '/Users/s1101153/Dropbox/Emily/Graphs/circularity/'
saveFile = savePath+time+'_'+min_flag+'_phip1'
pl.savefig(saveFile)
pl.show()

circ_data05.plot()
pl.xlabel('phi_r')
pl.ylabel('circularity measure')
if min_flag == 'abs':
    pl.title('phi_p = 0.5, minimum = single smallest value')
elif min_flag == 'ave':
    pl.title('phi_p = 0.5, minimum = average of smallest 10%')
time = datetime.now().strftime('%Y-%m-%d_%H%M')
savePath = '/Users/s1101153/Dropbox/Emily/Graphs/circularity/'
saveFile = savePath+time+'_'+min_flag+'_phip0-5'
pl.savefig(saveFile)
pl.show()
