import argparse

import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from random import random, uniform

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc
import pandas as pd

from models import *
import matplotlib.pylab as plt

parser = argparse.ArgumentParser()
parser.add_argument('sector', help='sector', type=str)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True

data = np.load('../model_input/data_q/sector'+args.sector+'.npz')

global_view = data['global_view']#[:,:3]
local_view = data['local_view']#[:,:3]
secondary_view = data['secondary_view']#[:,:3]
scalar = data['scalar']
tic = data['tic']
tce = data['tce']

scalar_mean_0 = 5.216795473250833
scalar_std_0 = 8.256681757964786
scalar_mean_1 = 4.5273240606418375
scalar_std_1 = 4.55454166155911
scalar_mean_2 = 5890.859165863254
scalar_std_2 = 1538.115397965016
scalar_mean_3 = 2.6935281168261334
scalar_std_3 = 8.169359116674578
scalar_mean_4 = 4.168138988323937
scalar_std_4 = 0.7169815866486623
scalar_mean_5 = 2.524581765093486
scalar_std_5 = 9.133970445454041
## these values will be used in the inferring process
scalar[:,0] -= scalar_mean_0
scalar[:,0] /= scalar_std_0
scalar[:,1] -= scalar_mean_1
scalar[:,1] /= scalar_std_1
scalar[:,2] -= scalar_mean_2
scalar[:,2] /= scalar_std_2
scalar[:,3] -= scalar_mean_3
scalar[:,3] /= scalar_std_3
scalar[:,4] -= scalar_mean_4
scalar[:,4] /= scalar_std_4
scalar[:,5] -= scalar_mean_5
scalar[:,5] /= scalar_std_5

global_view = np.clip(global_view, a_min=-5, a_max=5)
local_view = np.clip(local_view, a_min=-5, a_max=5)
secondary_view = np.clip(secondary_view, a_min=-5, a_max=5)
scalar = np.clip(scalar, a_min=-5, a_max=5)

### standardize scalar parameters
## mean and standard deviation calculated with the following commented lines
##for idx in range(6):
##    print('scalar_mean_'+str(idx),'=', np.mean(scalar[:,idx]))
##    print('scalar_std_'+str(idx),'=', np.std(scalar[:,idx]))


sector = data_utils.TensorDataset(torch.from_numpy(global_view).float(), torch.from_numpy(local_view).float(), \
                                  torch.from_numpy(secondary_view).float(), torch.from_numpy(scalar).float())

sectorloader = data_utils.DataLoader(sector, batch_size=64, shuffle=False)

net = Model(ch_in=3, n=32).cuda() ## convolutional network

for k in range(10):
    for t in ('a', 'b', 'c', 'd', 'e'):
        net.load_state_dict(torch.load('./output/model_q_'+t+'_'+str(k)+'.pt'))
        net.eval()
        idx = 0
        for (gv, lv, sv, sc) in sectorloader:
            gv, lv, sv, sc = gv.cuda(), lv.cuda(), sv.cuda(), sc.cuda()
            outputs = net(gv, lv, sv, sc)
            if idx == 0:
                prediction = softmax(outputs).data.cpu().numpy()
            else:
                prediction = np.concatenate((prediction, softmax(outputs).data.cpu().numpy()), axis=0)
            idx += 1
        if k==0 and t=='a':
            averaged_prediction = prediction
        else:
            averaged_prediction += prediction

        #print(prediction[:5])
        #plt.hist(prediction[:,1],bins=50)
        #plt.show()
        #plt.close()
        #ps = input()

averaged_prediction = averaged_prediction/50.0

#print(averaged_prediction[:5])

df = pd.DataFrame({'ticid': tic,
                   'tceid': tce,
                   'EB_probability': averaged_prediction[:,1],
                   'PC_probability': averaged_prediction[:,2]})

#sector = pd.read_csv('../tess_lc_download_sh/tess_tce_csv/'+csv_files[int(args.sector)], header=6)

#sector['EB_probability'] = averaged_prediction[:,1]
#sector['PC_probability'] = averaged_prediction[:,2]

df.to_csv('../predictions/tce_eb_pc_probability_sector_'+args.sector+'.csv')

print('inferring sector ',args.sector,' complete.')

#sector.to_csv('../predictions/tce_eb_pc_probability_sector_'+args.sector+'.csv', index_label='index', columns=['ticid', 'tceid', 'EB_probability', 'PC_probability'])

'''
csv_files = ['tess2018206190142-s0001-s0001_dvr-tcestats.csv',
             'tess2018235142541-s0002-s0002_dvr-tcestats.csv',
             'tess2018263124740-s0003-s0003_dvr-tcestats.csv',
             'tess2018292093539-s0004-s0004_dvr-tcestats.csv',
             'tess2018319112538-s0005-s0005_dvr-tcestats.csv',
             'tess2018349182739-s0006-s0006_dvr-tcestats.csv',  
             'tess2019008025936-s0007-s0007_dvr-tcestats.csv',  
             'tess2019033200935-s0008-s0008_dvr-tcestats.csv',  
             'tess2019059170935-s0009-s0009_dvr-tcestats.csv',  
             'tess2019085221934-s0010-s0010_dvr-tcestats.csv', 
             'tess2019113062933-s0011-s0011_dvr-tcestats.csv',
             'tess2019141104532-s0012-s0012_dvr-tcestats.csv',
             'tess2019170095531-s0013-s0013_dvr-tcestats.csv',
             'tess2019199201929-s0014-s0014_dvr-tcestats.csv',
             'tess2019227203528-s0015-s0015_dvr-tcestats.csv',
             'tess2019255032927-s0016-s0016_dvr-tcestats.csv',
             'tess2019281041526-s0017-s0017_dvr-tcestats.csv',
             'tess2019307033525-s0018-s0018_dvr-tcestats.csv',
             'tess2019332134924-s0019-s0019_dvr-tcestats.csv',
             'tess2019358235523-s0020-s0020_dvr-tcestats.csv',             
             'tess2020021221522-s0021-s0021_dvr-tcestats.csv',
             'tess2020050191121-s0022-s0022_dvr-tcestats.csv',
             'tess2020079142120-s0023-s0023_dvr-tcestats.csv',
             'tess2020107065519-s0024-s0024_dvr-tcestats.csv',
             'tess2020135030118-s0025-s0025_dvr-tcestats.csv',
             'tess2020161181517-s0026-s0026_dvr-tcestats.csv',
             'tess2020187183116-s0027-s0027_dvr-tcestats.csv',
             'tess2020213081515-s0028-s0028_dvr-tcestats.csv',
             'tess2020239173514-s0029-s0029_dvr-tcestats.csv',
             'tess2020267090513-s0030-s0030_dvr-tcestats.csv',
             'tess2020296001112-s0031-s0031_dvr-tcestats.csv',
             'tess2020325171311-s0032-s0032_dvr-tcestats.csv',
             'tess2020353052510-s0033-s0033_dvr-tcestats.csv',
             'tess2021014055109-s0034-s0034_dvr-tcestats.csv',
             'tess2021040113508-s0035-s0035_dvr-tcestats.csv',
             'tess2021066093107-s0036-s0036_dvr-tcestats.csv',
             'tess2021092173506-s0037-s0037_dvr-tcestats.csv',
             'tess2021119082105-s0038-s0038_dvr-tcestats.csv',
             'tess2021147062104-s0039-s0039_dvr-tcestats.csv']''' 
