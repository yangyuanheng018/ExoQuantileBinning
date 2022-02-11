import argparse

import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from random import random, uniform

from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc
parser = argparse.ArgumentParser()
#parser.add_argument('aug', help='data augmentation method', type=str)
parser.add_argument('data', help='data', type=str)
parser.add_argument('k', help='the k-th fold', type=int)

args = parser.parse_args()


## load the preprocessed training-validation date from the npz file
data = np.load('../model_input/training_data.npz')

dispositions = data['dispositions']

group = np.load('../model_input/training_dice2.npy')

test_group = (args.k+1)%10

test_indices = (group==test_group)

test_eb_label = dispositions[test_indices,1]
test_pc_label = dispositions[test_indices,2]

for t in ('a', 'b', 'c', 'd', 'e'):
    if t == 'a':
        prediction = np.load('./output/prediction_'+args.data+'_'+str(args.k)+'_'+t+'.npy')
    else:
        prediction += np.load('./output/prediction_'+args.data+'_'+str(args.k)+'_'+t+'.npy')

prediction /= 5.0
eb_aps_test = average_precision_score(test_eb_label, prediction[:,1])
pc_aps_test = average_precision_score(test_pc_label, prediction[:,2])
eb_auc_test = roc_auc_score(test_eb_label, prediction[:,1])
pc_auc_test = roc_auc_score(test_pc_label, prediction[:,2])

#print(prediction[:10])
print('ensembled results: pc aps, eb aps')
print(pc_aps_test, eb_aps_test)

'''
## OLD WAY
pred_test = [] ## model prediction and ground truth for the test set
for (gv, lv, sv, psd, targets) in testloader:
    gv, lv, sv, psd = gv.cuda(), lv.cuda(), sv.cuda(), psd.cuda()
    targets = targets.cuda()
    outputs = net(gv, lv, sv)
    pred_test.append(outputs.data.cpu().numpy())
    
pred_test = np.concatenate(pred_test).flatten()

targets = test_t.flatten()'''

results_txt = open('./output/results_'+args.data+'_'+str(args.k)+'_'+'.txt', 'w')
results_txt.write('EB_AUC '+str(eb_auc_test))
results_txt.write('\nEB_APS '+str(eb_aps_test))
results_txt.write('\nPC_AUC '+str(pc_auc_test))
results_txt.write('\nPC_APS '+str(pc_aps_test))

print('PC:')
results_txt.write('\n\n***** PC results: ******')
for threshold in (0.25,0.5):
    y_binarized = np.array(prediction[:,2] > threshold, dtype=float)
    print('PC: threshold:', threshold,'recovered: ', np.sum(y_binarized*test_pc_label), '/', np.sum(test_pc_label), \
          'precision {0:.4f}'.format(precision_score(test_pc_label, y_binarized, zero_division=1)), \
          'recall {0:.4f}'.format(recall_score(test_pc_label, y_binarized)))

    results_txt.write('\n\nthreshold: '+str(threshold)+'\nrecovered: '+str(np.sum(y_binarized*test_pc_label))+ '/'+str( np.sum(test_pc_label))+ \
                      '\nprecision {0:.4f}'.format(precision_score(test_pc_label, y_binarized, zero_division=1))+ \
                      '\nrecall {0:.4f}'.format(recall_score(test_pc_label, y_binarized)))




#print(prediction[:10])
#prediction = np.exp(prediction)
## EB results
print('EB:')
results_txt.write('\n\n****** EB results: ******')
for threshold in (0.25, 0.5):
    y_binarized = np.array(prediction[:,1] > threshold, dtype=float)
    print('threshold: ', threshold,' recovered: ', np.sum(y_binarized*test_eb_label), '/', np.sum(test_eb_label), \
          ' precision: {0:.4f}'.format(precision_score(test_eb_label, y_binarized, zero_division=1)), \
          ' recall: {0:.4f} '.format(recall_score(test_eb_label, y_binarized)))

    results_txt.write('\n\nthreshold: '+str(threshold)+'\nrecovered: '+str(np.sum(y_binarized*test_eb_label))+ '/'+str( np.sum(test_eb_label))+ \
                      '\nprecision: {0:.4f}'.format(precision_score(test_eb_label, y_binarized, zero_division=1))+ \
                      '\nrecall: {0:.4f}'.format(recall_score(test_eb_label, y_binarized)))
    ## PC results
results_txt.close()
