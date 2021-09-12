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


parser = argparse.ArgumentParser()
parser.add_argument('sector', help='sector', type=str)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True

sector_files = np.load('../model_input/data_q/sector'+args.sector+'.npz')
sector_lv = sector_files['local_view']
sector_gv = sector_files['global_view']
sector_sv = sector_files['secondary_view']

#print('sector local view max:',np.max(sector_lv))
#print('sector local view min:',np.min(sector_lv))
#print('sector local view max:',np.max(sector_gv))
#print('sector local view min:',np.min(sector_gv))

sector = data_utils.TensorDataset(torch.from_numpy(sector_gv).float(), torch.from_numpy(sector_lv).float(), torch.from_numpy(sector_sv).float())

sectorloader = data_utils.DataLoader(sector, batch_size=64, shuffle=False)

net = Model(ch_in=3, n=32).cuda() ## convolutional network

averaged_results = np.zeros(len(sector_lv))

for k in range(10):
    net.load_state_dict(torch.load('./output/model_q_i_'+str(k)+'.pt'))
    net.eval()

    pred_sector = [] ## model prediction and ground truth for the test set
    for (gv, lv, sv) in sectorloader:
        gv, lv, sv = gv.cuda(), lv.cuda(), sv.cuda()
        outputs = net(gv, lv, sv)
        pred_sector.append(outputs.data.cpu().numpy())
    
    pred_sector = np.concatenate(pred_sector).flatten()
    #print(pred_sector)
    averaged_results += pred_sector
    #print(averaged_results)
    #pz = input()

averaged_results = averaged_results/10.0

csv_files = ['tess2018349182739-s0006-s0006_dvr-tcestats.csv',  
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
             'tess2021147062104-s0039-s0039_dvr-tcestats.csv']

sector = pd.read_csv('../tess_lc_download_sh/tess_tce_csv/'+csv_files[int(args.sector)-6], header=6)

sector['probability'] = averaged_results


sector.to_csv('../tce_probability/tce_exopplanet_probability_sector'+args.sector+'.csv', index_label='index', columns=['ticid', 'tceid', 'probability'])
