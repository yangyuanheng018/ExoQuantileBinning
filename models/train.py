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

from models import *
## CNN model do not include scalar as input
#from cnn_model import *
    
def invert(tensor):
    idx = [i for i in range(tensor.size(2)-1, -1, -1)]
    idx = torch.cuda.LongTensor(idx)
    inverted_tensor = tensor.index_select(2, idx)
    return inverted_tensor

def transform(tensor, order):
    return torch.pow(torch.abs(tensor), order) * torch.sign(tensor)



parser = argparse.ArgumentParser()
#parser.add_argument('aug', help='data augmentation method', type=str)
parser.add_argument('data', help='data', type=str)
parser.add_argument('k', help='the k-th fold', type=int)
parser.add_argument('t', help='t-th experiments', type=str)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True


## load the preprocessed training-validation date from the npz file
data = np.load('../model_input/training_data.npz')

if 'q' in args.data:
    global_view = data['global_view'][:,:3]
    local_view = data['local_view'][:,:3]
    secondary_view = data['secondary_view'][:,:3]
else:
    global_view = data['global_view'][:,1:2]
    local_view = data['local_view'][:,1:2]
    secondary_view = data['secondary_view'][:,1:2]

scalar = data['scalar']

dispositions = data['dispositions']
print(dispositions.shape)

#dispositions = np.load('raw_disposition.npy')
#print(dispositions.shape)


#ps = input()
group = np.load('../model_input/training_dice2.npy')
#assert(group.shape[0] != dispositions.shape[0])
    
### standardize scalar parameters
## mean and standard deviation calculated with the following commented lines
##for idx in range(6):
##    print('scalar_mean_'+str(idx),'=', np.mean(scalar[:,idx]))
##    print('scalar_std_'+str(idx),'=', np.std(scalar[:,idx]))

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

target = dispositions[:,1] + 2*dispositions[:,2] ## 0 for junk; 1 for EB; 2 for PC
target = target.reshape(-1)
#print(global_view.shape)
print(target.shape)
#print(target[:2000:10])

#ps = input()


valid_group = args.k
test_group = (args.k+1)%10

valid_indices = (group==valid_group)
test_indices = (group==test_group)
train_indices = (np.logical_and(group!=valid_group, group!=test_group))

print('number of training data', np.sum(train_indices))
print('number of validation data', np.sum(valid_indices))
print('number of test data', np.sum(test_indices))

val_gv = global_view[valid_indices]
val_lv = local_view[valid_indices]
val_sv = secondary_view[valid_indices]
val_sc = scalar[valid_indices]
val_t = target[valid_indices]

test_gv = global_view[test_indices]
test_lv = local_view[test_indices]
test_sv = secondary_view[test_indices]
test_sc = scalar[test_indices]
test_t = target[test_indices]

train_gv = global_view[train_indices]
train_lv = local_view[train_indices]
train_sv = secondary_view[train_indices]
train_sc = scalar[train_indices]
train_t = target[train_indices]

## EB and PC label, to calculate APS and AUC etc
valid_eb_label = dispositions[valid_indices,1]
valid_pc_label = dispositions[valid_indices,2]

test_eb_label = dispositions[test_indices,1]
test_pc_label = dispositions[test_indices,2]

print(val_t.shape)
print(np.sum(valid_eb_label), np.sum(valid_pc_label), np.sum(test_eb_label), np.sum(test_pc_label))

print(train_gv.shape, train_lv.shape, train_sv.shape, train_sc.shape, train_t.shape)
print(val_gv.shape, val_lv.shape, val_sv.shape, val_sc.shape, val_t.shape)
print(test_gv.shape, test_lv.shape, test_sv.shape, test_sc.shape, test_t.shape)

'''
epsilon = 1e-6
for indx in range(len(train_gv)):
    flux_median = np.median(train_gv[indx,1])
    train_gv[indx] = train_gv[indx] - flux_median
    flux_median = np.median(train_lv[indx,1])
    train_lv[indx] = train_lv[indx] - flux_median

    flux_min = abs(np.min(train_gv[indx,1]))
    train_gv[indx] = train_gv[indx] / (flux_min + epsilon)
    flux_min = abs(np.min(train_lv[indx,1]))
    train_lv[indx] = train_lv[indx] / (flux_min + epsilon)

for indx in range(len(val_gv)):
    flux_median = np.median(val_gv[indx,1])
    val_gv[indx] = val_gv[indx] - flux_median
    flux_median = np.median(val_lv[indx,1])
    val_lv[indx] = val_lv[indx] - flux_median

    flux_min = abs(np.min(val_gv[indx,1]))
    val_gv[indx] = val_gv[indx] / (flux_min + epsilon)
    flux_min = abs(np.min(val_lv[indx,1]))
    val_lv[indx] = val_lv[indx] / (flux_min + epsilon)

for indx in range(len(test_gv)):
    flux_median = np.median(test_gv[indx,1])
    test_gv[indx] = test_gv[indx] - flux_median
    flux_median = np.median(test_lv[indx,1])
    test_lv[indx] = test_lv[indx] - flux_median

    flux_min = abs(np.min(test_gv[indx,1]))
    test_gv[indx] = test_gv[indx] / (flux_min + epsilon)
    flux_min = abs(np.min(test_lv[indx,1]))
    test_lv[indx] = test_lv[indx] / (flux_min + epsilon)'''

#train_gv = np.clip(train_gv, a_min=-1.5, a_max=3.0)
#val_gv = np.clip(val_gv, a_min=-1.5, a_max=3.0)
#test_gv = np.clip(test_gv, a_min=-1.5, a_max=3.0)
#train_lv = np.clip(train_lv, a_min=-1.5, a_max=3.0)
#val_lv = np.clip(val_lv, a_min=-1.5, a_max=3.0)
#test_lv = np.clip(test_lv, a_min=-1.5, a_max=3.0)

'''
n_global_bins = 201
n_local_bins = 61
for indx in range(len(train_gv)):
    if abs(np.max(train_lv[indx])) > 100 or abs(np.min(train_lv[indx])) > 100 or \
       abs(np.max(train_gv[indx])) > 100 or abs(np.min(train_gv[indx])) > 100:
        figure, axis = plt.subplots(2, 1)
        axis[0].set_title(train_ds[indx] + ' Global view')
        axis[1].set_title('Local view')
        axis[0].plot(np.arange(n_global_bins), train_gv[indx, 0], 'b.')
        axis[0].plot(np.arange(n_global_bins), train_gv[indx, 1], 'r.')
        axis[0].plot(np.arange(n_global_bins), train_gv[indx, 2], 'g.')
        axis[1].plot(np.arange(n_local_bins), train_lv[indx, 0], 'b.')
        axis[1].plot(np.arange(n_local_bins), train_lv[indx, 1], 'r.')
        axis[1].plot(np.arange(n_local_bins), train_lv[indx, 2], 'g.')
        plt.show()
        plt.close()
'''



#train = data_utils.TensorDataset(torch.from_numpy(train_gv).float(), torch.from_numpy(train_lv).float(), torch.from_numpy(train_t).float())
#val = data_utils.TensorDataset(torch.from_numpy(val_gv).float(), torch.from_numpy(val_lv).float(), torch.from_numpy(val_t).float())
#test = data_utils.TensorDataset(torch.from_numpy(test_gv).float(), torch.from_numpy(test_lv).float(), torch.from_numpy(test_t).float())

train = data_utils.TensorDataset(torch.from_numpy(train_gv).float(), torch.from_numpy(train_lv).float(), torch.from_numpy(train_sv).float(),\
                                 torch.from_numpy(train_sc).float(), torch.from_numpy(train_t).long())
val = data_utils.TensorDataset(torch.from_numpy(val_gv).float(), torch.from_numpy(val_lv).float(), torch.from_numpy(val_sv).float(),\
                               torch.from_numpy(val_sc).float(), torch.from_numpy(val_t).long())
test = data_utils.TensorDataset(torch.from_numpy(test_gv).float(), torch.from_numpy(test_lv).float(), torch.from_numpy(test_sv).float(),\
                                torch.from_numpy(test_sc).float(), torch.from_numpy(test_t).long())

trainloader = data_utils.DataLoader(train, batch_size=64, shuffle=True)
valloader = data_utils.DataLoader(val, batch_size=64, shuffle=False)
testloader = data_utils.DataLoader(test, batch_size=64, shuffle=False)

if 'q' in args.data:
    net = Model(ch_in=3, n=32).cuda() ## plain convolutional network
else:
    net = Model(ch_in=1, n=32).cuda() ## plain convolutional network

## count number of parameters in this neural network model
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('This model has', total_params, 'parameters.')

criterion = nn.CrossEntropyLoss().cuda()
#criterion = nn.MSELoss().cuda()
#criterion = nn.NLLLoss()
#nn.BCELoss()

optimizer = optim.Adam(net.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8)

train_log = open('./output/training_record_'+args.data+'_'+args.t+'_'+str(args.k)+'.log','w')
train_log.write('epoch,train_loss,test_loss,test_aps,test_auc\n')

print('start training, data: '+str(args.data)+' augmentation: '+str(args.t)+' experiment: '+str(args.k))
best_loss_val = 200.0
best_pc_aps_valid = 0.0
tolerence = 0

print('epoch,loss_train, loss_val,loss_test | eb_aps_valid, eb_aps_test | eb_auc_valid, eb_auc_test || pc_aps_valid, pc_aps_test, pc_auc_valid, pc_auc_test')

for epoch in range(1,151,1):
    net.train()
    loss_train = 0.0
    #num_corr_train, num_train = 0.0, 0.0
    for (gv, lv, sv, sc, targets) in trainloader:
        gv, lv, sv, sc, targets = gv.cuda(), lv.cuda(), sv.cuda(), sc.cuda(), targets.cuda()
        gv = invert(gv)
        lv = invert(lv)
        sv = invert(sv)

        '''
        if 'i' in args.aug and random()<0.5: ## 50 percent possibility to invert time
            gv = invert(gv)
            lv = invert(lv)
            sv = invert(sv)

        if 'f' in args.aug:
            order = uniform(2.0,0.5)
            gv = transform(gv, order)
            lv = transform(lv, order)
            sv = transform(sv, order)'''
        
        outputs = net(gv, lv, sv, sc)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_train += loss.item()*targets.size(0)

    ## results on the validataion set
    net.eval()

    loss_val = 0.0
    idx = 0 ## for the first mini-batch, do not cat; for the rest, cat with the previous
    for (gv, lv, sv, sc, targets) in valloader:
        gv, lv, sv, sc, targets = gv.cuda(), lv.cuda(), sv.cuda(), sc.cuda(), targets.cuda()
        outputs = net(gv, lv, sv, sc)
        loss = criterion(outputs, targets)
        loss_val += loss.item()*targets.size(0)
        if idx == 0:
            prediction = softmax(outputs).data.cpu().numpy()
        else:
            prediction = np.concatenate((prediction, softmax(outputs).data.cpu().numpy()), axis=0)
        idx += 1
    #prediction = np.exp(prediction)
    eb_aps_valid = average_precision_score(valid_eb_label, prediction[:,1])
    pc_aps_valid = average_precision_score(valid_pc_label, prediction[:,2])
    eb_auc_valid = roc_auc_score(valid_eb_label, prediction[:,1])
    pc_auc_valid = roc_auc_score(valid_pc_label, prediction[:,2])

    ## results on the test set
    loss_test = 0.0
    idx = 0
    for (gv, lv, sv, sc, targets) in testloader:
        gv, lv, sv, sc, targets = gv.cuda(), lv.cuda(), sv.cuda(), sc.cuda(), targets.cuda()
        outputs = net(gv, lv, sv, sc)
        loss = criterion(outputs, targets)
        loss_test += loss.item()*targets.size(0)
        if idx == 0:
            prediction = softmax(outputs).data.cpu().numpy()
        else:
            prediction = np.concatenate((prediction, softmax(outputs).data.cpu().numpy()), axis=0)
        idx += 1
    #prediction = np.exp(prediction)
    eb_aps_test = average_precision_score(test_eb_label, prediction[:,1])
    pc_aps_test = average_precision_score(test_pc_label, prediction[:,2])
    eb_auc_test = roc_auc_score(test_eb_label, prediction[:,1])
    pc_auc_test = roc_auc_score(test_pc_label, prediction[:,2])

    ## print current results on the screen
    print(epoch, ' {0:.1f}'.format(loss_train), ' {0:.1f}'.format(loss_val),' {0:.1f}'.format(loss_test),\
          '| {0:.4f}'.format(eb_aps_valid), ' {0:.4f}'.format(eb_aps_test), \
          '| {0:.4f}'.format(eb_auc_valid), ' {0:.4f}'.format(eb_auc_test), \
          '|| {0:.4f}'.format(pc_aps_valid), ' {0:.4f}'.format(pc_aps_test), \
          '| {0:.4f}'.format(pc_auc_valid), ' {0:.4f}'.format(pc_auc_test), end=' | ')
    if loss_val < best_loss_val:
        #if pc_aps_valid > best_pc_aps_valid:
        best_loss_val = loss_val
        #best_pc_aps_valid = pc_aps_valid
        torch.save(net.state_dict(), './output/model_'+args.data+'_'+str(args.t)+'_'+str(args.k)+'.pt')
        tolerence = 0
        print('model saved...')
    else:
        print(' ')
    tolerence += 1
    if tolerence>30:
        break


    train_log.write('{:d},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n'\
                    .format(epoch, loss_train, loss_test, eb_aps_test, eb_auc_test, pc_aps_test, pc_auc_test))

train_log.close()

#######################
## training finished ##
#######################

## load the model parameters with the best performance on the validation set
net.load_state_dict(torch.load('./output/model_'+args.data+'_'+str(args.t)+'_'+str(args.k)+'.pt'))
net.eval()

idx = 0
for (gv, lv, sv, sc, targets) in testloader:
    gv, lv, sv, sc, targets = gv.cuda(), lv.cuda(), sv.cuda(), sc.cuda(), targets.cuda()
    outputs = net(gv, lv, sv, sc)
    if idx == 0:
        prediction = softmax(outputs).data.cpu().numpy()
    else:
        prediction = np.concatenate((prediction, softmax(outputs).data.cpu().numpy()), axis=0)
    idx += 1
    
eb_aps_test = average_precision_score(test_eb_label, prediction[:,1])
pc_aps_test = average_precision_score(test_pc_label, prediction[:,2])
eb_auc_test = roc_auc_score(test_eb_label, prediction[:,1])
pc_auc_test = roc_auc_score(test_pc_label, prediction[:,2])

np.save('./output/prediction_'+args.data+'_'+str(args.k)+'_'+str(args.t)+'.npy', prediction)

#print(prediction[:10])
print(eb_aps_test, pc_aps_test, eb_auc_test, pc_auc_test)

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

results_txt = open('./output/results_'+args.data+'_'+args.t+'_'+str(args.k)+'.txt','w')
results_txt.write('EB_AUC '+str(eb_auc_test))
results_txt.write('\nEB_APS '+str(eb_aps_test))
results_txt.write('\nPC_AUC '+str(pc_auc_test))
results_txt.write('\nPC_APS '+str(pc_aps_test))

#print(prediction[:10])
#prediction = np.exp(prediction)
## EB results
print('EB:')
for threshold in (0.25, 0.5, 0.75):
    y_binarized = np.array(prediction[:,1] > threshold, dtype=float)
    print('threshold: ', threshold,' recovered', np.sum(y_binarized*test_eb_label), '/', np.sum(test_eb_label), \
          ' precision: {0:.4f}'.format(precision_score(test_eb_label, y_binarized, zero_division=1)), \
          ' recall: {0:.4f} '.format(recall_score(test_eb_label, y_binarized)))

    results_txt.write('\nEB results: threshold: '+str(threshold)+'\nrecovered'+str(np.sum(y_binarized*test_eb_label))+ '/'+str( np.sum(test_eb_label))+ \
          '\nprecision: {0:.4f}'.format(precision_score(test_eb_label, y_binarized, zero_division=1))+ \
          '\nrecall: {0:.4f}\n'.format(recall_score(test_eb_label, y_binarized)))
## PC results
print('PC:')
for threshold in (0.25, 0.5, 0.75):
    y_binarized = np.array(prediction[:,2] > threshold, dtype=float)
    print('PC: threshold:', threshold,'recovered', np.sum(y_binarized*test_pc_label), '/', np.sum(test_pc_label), \
          'precision {0:.4f}'.format(precision_score(test_pc_label, y_binarized, zero_division=1)), \
          'recall {0:.4f}'.format(recall_score(test_pc_label, y_binarized)))

    results_txt.write('\nPC results: threshold'+str(threshold)+'\nrecovered'+str(np.sum(y_binarized*test_eb_label))+ '/'+str( np.sum(test_eb_label))+ \
          '\nprecision {0:.4f}'.format(precision_score(test_eb_label, y_binarized, zero_division=1))+ \
          '\nrecall {0:.4f}'.format(recall_score(test_eb_label, y_binarized)))

np.save('./output/test_prediction_'+args.data+'_'+str(args.t)+'_'+str(args.k)+'.npy', prediction)

del net

'''

print('start training model '+str(args.model)+' augmentation '+str(args.aug)+' fold '+str(args.k))
best_loss_val = 200.0
tolerence = 0
for epoch in range(1,201,1):
    net.train()
    loss_train = 0.0
    #num_corr_train, num_train = 0.0, 0.0
    for (_, x1, x2, x3, targets) in trainloader:

        if 'i' in args.aug:
            if random()<0.5:
                flc = invert(flc)
        if 'r' in args.aug:
            if random()<0.9:
                cut = np.random.randint(-1000,1000)
                flc = hscale(flc, 10039, cut)
        if 's' in args.aug:
            if random()<0.9:
                inputs = channel_shuffle(flc)
        
        outputs = net(x1,x2,x3)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_train += loss.item()*targets.size(0)

    ## results on the validataion set
    net.eval()

    loss_val = 0.0
    prediction, groundtruth = [], []
    for (_, x1, x2, x3, targets) in valloader:

        outputs = net(x1,x2,x3)
        loss = criterion(outputs, targets)
        loss_val += loss.item()*targets.size(0)
        prediction.append(outputs.data.cpu().numpy())
        groundtruth.append(targets.data.cpu().numpy())
    prediction = np.concatenate(prediction).flatten()
    groundtruth = np.concatenate(groundtruth).flatten()
    aps_val = average_precision_score(groundtruth, prediction)
    auc_val = roc_auc_score(groundtruth, prediction)

    loss_test = 0.0
    prediction, groundtruth = [], []
    for (_, x1, x2, x3, targets) in testloader:

        outputs = net(x1,x2,x3)
        loss = criterion(outputs, targets)
        loss_test += loss.item()*targets.size(0)
        prediction.append(outputs.data.cpu().numpy())
        groundtruth.append(targets.data.cpu().numpy())
    prediction = np.concatenate(prediction).flatten()
    groundtruth = np.concatenate(groundtruth).flatten()
           
    aps_test = average_precision_score(groundtruth, prediction)
    auc_test = roc_auc_score(groundtruth, prediction)


    print(epoch, ' {0:.1f}'.format(loss_train), ' {0:.1f}'.format(loss_val),' {0:.1f}'.format(loss_test),\
          '| {0:.4f}'.format(aps_val), \
          ' {0:.4f}'.format(aps_test), \
          '| {0:.4f}'.format(auc_val), \
          ' {0:.4f}'.format(auc_test))

    if loss_val < best_loss_val:
        best_loss_val = loss_val
        torch.save(net.state_dict(), './output/model'+str(args.model)+'_'+str(args.aug)+'_'+str(args.k)+'.pt')
        print('model saved...')
        tolerence = 0
    tolerence += 1
    if tolerence>30:
        break

    train_log.write('{:d},{:.5f},{:.5f},{:.5f},{:.5f}\n'\
                    .format(epoch, loss_train, loss_test, aps_test, auc_test))

train_log.close()

#######################
## training finished ##
#######################

## load the model parameters with the best performance on the validation set
net.load_state_dict(torch.load('./output/model'+str(args.model)+'_'+str(args.aug)+'_'+str(args.k)+'.pt'))

one_test_loader = data_utils.DataLoader(test, batch_size=1, shuffle=False)
net.eval()
idx = 0
for (ticid, x1, x2, x3, targets) in one_test_loader:

    outputs = net(x1,x2,x3)
    if abs(outputs.item() - targets.item()) > 0.5:
        time_length = 10039
        channel_fold = inputs[0]
        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, top=0.95, right=0.90, bottom=0.05, wspace=0.15, hspace=0.35)
        plt.subplot(7, 1, 1)
        plt.title('TIC: {}  Disposition: {} Prediction: {}'.format(ticid, targets.item(), np.round(outputs.item())))
        idx = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
        for i, idx_c in enumerate(idx):
            ax = plt.subplot(7, 2, i + 3)
            if i < 10:
                ax.xaxis.set_ticklabels([])
            ax.set(xlim=(0, time_length), ylim=(-1., 1.0), yticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
            # ax.scatter(np.arange(time_length), channel_fold[idx_c], c='lavender', cmap='RGB', s=2, label='channel {}'.format(idx_c+1))
            ax.plot(np.arange(time_length), channel_fold[idx_c], '.', color='teal', markersize=2.5,
                    label='channel {}'.format(idx_c + 1))
            ax.legend(loc='lower right', fontsize='small', frameon=True)
        plt.savefig('../figure/misclass{}.png'.format(ticid))
        #plt.show()
        plt.close()
        
net.eval()
pred_test = [] ## model prediction and ground truth for the test set
for (ticid, x1, x2, x3, targets) in testloader:

    outputs = net(x1,x2,x3)
    pred_test.append(outputs.data.cpu().numpy())
pred_test = np.concatenate(pred_test).flatten()
np.save('./output/test_prediction_kfold_'+str(args.model)+'_'+str(args.aug)+'_'+str(args.k)+'.npy', pred_test)


## output the prediction for TOIs
tois = FLCDataset('../model_input/tois.npz', mode=args.mode)
toiloader = DataLoader(tois, shuffle=False, num_workers=1, batch_size=64)

pred_toi = [] ## model prediction and ground truth for the toi set
for (ticid, x1, x2, x3, targets) in toiloader:

    outputs = net(x1, x2, x3)
    pred_toi.append(outputs.data.cpu().numpy())
    
pred_toi = np.concatenate(pred_toi).flatten()
np.save('./output/toi_prediction_kfold_'+str(args.model)+'_'+str(args.aug)+'_'+str(args.k)+'.npy', pred_toi)'''
