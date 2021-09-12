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

#for _ in range(10):
#    print(random.uniform(1.2,0.8))
#s = np.random.uniform(-1,0,1000)
#print(s)
#pz = input()



class Model(nn.Module):
    def __init__(self, ch_in=3, n=16):
        super(Model, self).__init__()
        self.conv_gv = nn.Sequential(
            nn.Conv1d(ch_in, n, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(n, n, 5, stride=1, padding=2),
            nn.ReLU(),
            
            nn.MaxPool1d(5, stride=2, return_indices=False),

            nn.Conv1d(n, 2*n, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(2*n, 2*n, 5, stride=1, padding=2),
            nn.ReLU(),

            nn.MaxPool1d(5, stride=2, return_indices=False),

            nn.Conv1d(2*n, 4*n, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(4*n, 4*n, 5, stride=1, padding=2),
            nn.ReLU(),
            
            nn.MaxPool1d(5, stride=2, return_indices=False),

            nn.Conv1d(4*n, 8*n, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(8*n, 8*n, 5, stride=1, padding=2),
            nn.ReLU(),

            nn.MaxPool1d(5, stride=2, return_indices=False),

            nn.Conv1d(8*n, 16*n, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16*n, 16*n, 5, stride=1, padding=2),
            nn.ReLU(),

            nn.MaxPool1d(5, stride=2, return_indices=False))

        self.conv_lv = nn.Sequential(
            nn.Conv1d(ch_in, n, 5, stride=1, padding=2),
            nn.ReLU(),
            #nn.Conv1d(n, n, 5, stride=1, padding=2),
            #nn.ReLU(),
            nn.Conv1d(n, n, 5, stride=1, padding=2),
            nn.ReLU(),
            
            nn.MaxPool1d(7, stride=2, return_indices=False),

            nn.Conv1d(n, 2*n, 5, stride=1, padding=2),
            nn.ReLU(),
            #nn.Conv1d(2*n, 2*n, 5, stride=1, padding=2),
            #nn.ReLU(),
            nn.Conv1d(2*n, 2*n, 5, stride=1, padding=2),
            nn.ReLU(),

            nn.MaxPool1d(7, stride=2, return_indices=False))

        self.conv_sv = nn.Sequential(
            nn.Conv1d(ch_in, n, 5, stride=1, padding=2),
            nn.ReLU(),
            #nn.Conv1d(n, n, 5, stride=1, padding=2),
            #nn.ReLU(),
            nn.Conv1d(n, n, 5, stride=1, padding=2),
            nn.ReLU(),
            
            nn.MaxPool1d(7, stride=2, return_indices=False),

            nn.Conv1d(n, 2*n, 5, stride=1, padding=2),
            nn.ReLU(),
            #nn.Conv1d(2*n, 2*n, 5, stride=1, padding=2),
            #nn.ReLU(),
            nn.Conv1d(2*n, 2*n, 5, stride=1, padding=2),
            nn.ReLU(),

            nn.MaxPool1d(7, stride=2, return_indices=False))

        self.linear = nn.Sequential(
            nn.Linear(92*n, 64*n),
            nn.ReLU(),
            #nn.Linear(32*n, 32*n),
            #nn.ReLU(),
            #nn.Linear(32*n, 32*n),
            #nn.ReLU(),
            #nn.Linear(32*n, 32*n),
            #nn.ReLU(),
            nn.Linear(64*n, 1),
            nn.Sigmoid())

        #self.drop = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, gv, lv, sv):
        gv_out = self.conv_gv(gv)
        lv_out = self.conv_lv(lv)
        sv_out = self.conv_lv(sv)
        gv_out = gv_out.view(gv_out.shape[0], -1)
        lv_out = lv_out.view(lv_out.shape[0], -1)
        sv_out = sv_out.view(sv_out.shape[0], -1)
        #print(gv_out.size(), lv_out.size(), psd.size())
        out = torch.cat((gv_out, lv_out, sv_out), dim=1)
        #out = self.drop(out)
        #print(out.size())
        out = self.linear(out)

        return out

def invert(tensor):
    idx = [i for i in range(tensor.size(2)-1, -1, -1)]
    idx = torch.cuda.LongTensor(idx)
    inverted_tensor = tensor.index_select(2, idx)
    return inverted_tensor

def transform(tensor, order):
    return torch.pow(torch.abs(tensor), order) * torch.sign(tensor)

parser = argparse.ArgumentParser()
parser.add_argument('data', help='data', type=str)
parser.add_argument('aug', help='data augmentation method', type=str)
parser.add_argument('k', help='the k-th fold', type=int)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True

## load pre-shuffled index order file to split train-validation-test data
dice = np.load('dice'+str(args.k)+'.npy')

train_files = np.load('../model_input/data_q/train.npz')
print(train_files.files)
if 'q' in args.data:
    train_lv = train_files['local_view']
    train_gv = train_files['global_view']
    train_sv = train_files['secondary_view']
else:
    train_lv = train_files['local_view'][:,1:2,:]
    train_gv = train_files['global_view'][:,1:2,:]
    train_sv = train_files['secondary_view'][:,1:2,:]

train_psd = train_files['psd'].reshape((-1,1))
train_ds = train_files['dispositions']

## train vetting label
pc = ['PC']*train_ds.shape[0]
train_t = np.array(train_ds == pc, dtype='int').reshape((len(train_ds),1))

val_files = np.load('../model_input/data_q/val.npz')
if 'q' in args.data:
    val_lv = val_files['local_view']
    val_gv = val_files['global_view']
    val_sv = val_files['secondary_view']
else:
    val_lv = val_files['local_view'][:,1:2,:]
    val_gv = val_files['global_view'][:,1:2,:]
    val_sv = val_files['secondary_view'][:,1:2,:]

val_psd = val_files['psd'].reshape((-1,1))
val_ds = val_files['dispositions']

pc = ['PC']*val_ds.shape[0]
val_t = np.array(val_ds == pc, dtype='int').reshape((len(val_ds),1))

test_files = np.load('../model_input/data_q/test.npz')
if 'q' in args.data:
    test_lv = test_files['local_view']
    test_gv = test_files['global_view']
    test_sv = test_files['secondary_view']
else:
    test_lv = test_files['local_view'][:,1:2,:]
    test_gv = test_files['global_view'][:,1:2,:]
    test_sv = test_files['secondary_view'][:,1:2,:]

test_psd = test_files['psd'].reshape((-1,1))
test_ds = test_files['dispositions']

pc = ['PC']*test_ds.shape[0]
test_t = np.array(test_ds == pc, dtype='int').reshape((len(test_ds),1))

## combine all data together and used the shuffled index to re-split train-val-test
all_lv = np.concatenate((train_lv, val_lv, test_lv))
all_gv = np.concatenate((train_gv, val_gv, test_gv))
all_sv = np.concatenate((train_sv, val_sv, test_sv))
all_psd = np.concatenate((train_psd, val_psd, test_psd))
all_t = np.concatenate((train_t, val_t, test_t))

train_lv = all_lv[dice[:-872]]
val_lv = all_lv[dice[-872:-436]]
test_lv = all_lv[dice[-436:]]

train_gv = all_gv[dice[:-872]]
val_gv = all_gv[dice[-872:-436]]
test_gv = all_gv[dice[-436:]]

train_sv = all_sv[dice[:-872]]
val_sv = all_sv[dice[-872:-436]]
test_sv = all_sv[dice[-436:]]

train_psd = all_psd[dice[:-872]]
val_psd = all_psd[dice[-872:-436]]
test_psd = all_psd[dice[-436:]]

train_t = all_t[dice[:-872]]
val_t = all_t[dice[-872:-436]]
test_t = all_t[dice[-436:]]


print(train_gv.shape, train_lv.shape, train_sv.shape, train_t.shape)
print(val_gv.shape, val_lv.shape, val_sv.shape, val_t.shape)
print(test_gv.shape, test_lv.shape, test_sv.shape, test_t.shape)
print(train_t.mean(), val_t.mean(), test_t.mean())


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

print('train local view max:',np.max(train_lv))
print('train local view min:',np.min(train_lv))
print('train local view max:',np.max(train_gv))
print('train local view min:',np.min(train_gv))

print('val local view max:',np.max(val_lv))
print('val local view min:',np.min(val_lv))
print('val local view max:',np.max(val_gv))
print('val local view min:',np.min(val_gv))

print('test local view max:',np.max(test_lv))
print('test local view min:',np.min(test_lv))
print('test local view max:',np.max(test_gv))
print('test local view min:',np.min(test_gv))


#train = data_utils.TensorDataset(torch.from_numpy(train_gv).float(), torch.from_numpy(train_lv).float(), torch.from_numpy(train_t).float())
#val = data_utils.TensorDataset(torch.from_numpy(val_gv).float(), torch.from_numpy(val_lv).float(), torch.from_numpy(val_t).float())
#test = data_utils.TensorDataset(torch.from_numpy(test_gv).float(), torch.from_numpy(test_lv).float(), torch.from_numpy(test_t).float())

train = data_utils.TensorDataset(torch.from_numpy(train_gv).float(), torch.from_numpy(train_lv).float(), torch.from_numpy(train_sv).float(),\
                                 torch.from_numpy(train_psd).float(), torch.from_numpy(train_t).float())
val = data_utils.TensorDataset(torch.from_numpy(val_gv).float(), torch.from_numpy(val_lv).float(), torch.from_numpy(val_sv).float(),\
                               torch.from_numpy(val_psd).float(), torch.from_numpy(val_t).float())
test = data_utils.TensorDataset(torch.from_numpy(test_gv).float(), torch.from_numpy(test_lv).float(), torch.from_numpy(test_sv).float(),\
                                torch.from_numpy(test_psd).float(), torch.from_numpy(test_t).float())

trainloader = data_utils.DataLoader(train, batch_size=64, shuffle=True)
valloader = data_utils.DataLoader(val, batch_size=64, shuffle=True)
testloader = data_utils.DataLoader(test, batch_size=64, shuffle=False)

if 'q' in args.data:
    net = Model(ch_in=3, n=32).cuda() ## plain convolutional network
else:
    net = Model(ch_in=1, n=32).cuda() ## plain convolutional network

## count number of parameters in this neural network model
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('This model has', total_params, 'parameters.')

criterion = nn.BCELoss()

optimizer = optim.Adam(net.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8)

train_log = open('./output/training_record_'+args.data+'_'+args.aug+'_'+str(args.k)+'.log','w')
train_log.write('epoch,train_loss,test_loss,test_aps,test_auc\n')

print('start training, data: '+str(args.data)+' augmentation: '+str(args.aug)+' experiment: '+str(args.k))
best_loss_val = 200.0
best_aps_val = 0.0
tolerence = 0
for epoch in range(1,101,1):
    net.train()
    loss_train = 0.0
    #num_corr_train, num_train = 0.0, 0.0
    for (gv, lv, sv, psd, targets) in trainloader:
        gv, lv, sv, psd = gv.cuda(), lv.cuda(), sv.cuda(), psd.cuda()
        targets = targets.cuda()

        if 'i' in args.aug and random()<0.5: ## 50 percent possibility to invert time
            gv = invert(gv)
            lv = invert(lv)
            sv = invert(sv)

        if 'f' in args.aug:
            order = uniform(2.0,0.5)
            gv = transform(gv, order)
            lv = transform(lv, order)
            sv = transform(sv, order)
        
        outputs = net(gv, lv, sv)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_train += loss.item()*targets.size(0)

    ## results on the validataion set
    net.eval()

    loss_val = 0.0
    prediction, groundtruth = [], []
    for (gv, lv, sv, psd, targets) in valloader:
        gv, lv, sv, psd = gv.cuda(), lv.cuda(), sv.cuda(), psd.cuda()
        targets = targets.cuda()
        outputs = net(gv, lv, sv)
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
    for (gv, lv, sv, psd, targets) in testloader:
        gv, lv, sv, psd = gv.cuda(), lv.cuda(), sv.cuda(), psd.cuda()
        targets = targets.cuda()
        outputs = net(gv, lv, sv)
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

    #if loss_val < best_loss_val:
    if aps_val > best_aps_val:
        #best_loss_val = loss_val
        best_aps_val = aps_val
        torch.save(net.state_dict(), './output/model_'+args.data+'_'+str(args.aug)+'_'+str(args.k)+'.pt')
        print('model saved...')
        tolerence = 0
    tolerence += 1
    if tolerence>20:
        break


    train_log.write('{:d},{:.5f},{:.5f},{:.5f},{:.5f}\n'\
                    .format(epoch, loss_train, loss_test, aps_test, auc_test))

train_log.close()

#######################
## training finished ##
#######################

## load the model parameters with the best performance on the validation set
net.load_state_dict(torch.load('./output/model_'+args.data+'_'+str(args.aug)+'_'+str(args.k)+'.pt'))
net.eval()

pred_test = [] ## model prediction and ground truth for the test set
for (gv, lv, sv, psd, targets) in testloader:
    gv, lv, sv, psd = gv.cuda(), lv.cuda(), sv.cuda(), psd.cuda()
    targets = targets.cuda()
    outputs = net(gv, lv, sv)
    pred_test.append(outputs.data.cpu().numpy())
    
pred_test = np.concatenate(pred_test).flatten()

targets = test_t.flatten()

results_txt = open('./output/test_result_'+args.data+'_'+args.aug+'_'+str(args.k)+'.txt','w')
results_txt.write('AUC '+str(roc_auc_score(targets, pred_test)))
results_txt.write('\nAPS '+str(average_precision_score(targets, pred_test)))

print('AUC ',roc_auc_score(targets, pred_test))
print('APS ', average_precision_score(targets, pred_test))

for threshold in (0.25, 0.5, 0.75):
    y_binarized = np.array(pred_test > threshold, dtype=float)
    print('threshold', threshold,'recovered', np.sum(y_binarized*targets), '/', np.sum(targets), \
          'precision {0:.4f}'.format(precision_score(targets, y_binarized, zero_division=1)), \
          'recall {0:.4f}'.format(recall_score(targets, y_binarized)))

    results_txt.write('\nthreshold'+str(threshold)+'recovered'+str(np.sum(y_binarized*targets))+'/'+ str(np.sum(targets))+ \
          'precision {0:.4f}'.format(precision_score(targets, y_binarized, zero_division=1))+ \
          'recall {0:.4f}'.format(recall_score(targets, y_binarized)))

np.save('./output/test_'+args.data+'_'+str(args.aug)+'_'+str(args.k)+'.npy', pred_test)


