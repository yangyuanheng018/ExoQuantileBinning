from astropy.io import fits
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal
#from wotan import flatten
import time as tm
import matplotlib.pyplot as plt

'''
start = tm.time()
x = np.random.rand(12)
print(x)
for _ in range(100000):
    #y = np.median(x)
    y = np.mean(x)
    #y = np.quantile(x,[0.25, 0.5,0.75])
print(tm.time()-start)
pz = input('time test over')'''


def seg_consecutive(x, gap_limit, length_limit):
    x_gap = x[1:] - x[:-1]
    segs = []
    inter_seg_left = x[0]
    for i in range(len(x_gap)):
        if x_gap[i]>gap_limit:
            segs.append([inter_seg_left, x[i]])
            inter_seg_left = x[i+1]
    segs.append([inter_seg_left, x[-1]])
    trimmed_seg = []
    for g in segs:
        if g[1]-g[0]>length_limit:
            trimmed_seg.append(g)
    #print(trimmed_seg)
    
    return trimmed_seg
    
def no_badtime_segupsample(time, flux, cadence, up_factor, gap_limit, length_limit, filled_value=1.0):
    segs = seg_consecutive(time, gap_limit, length_limit)
    inter_flux = interp1d(time, flux, kind='cubic')

    up_time, up_flux = [], []
    for i, seg in enumerate(segs):
        seg_time = np.linspace(seg[0], seg[1], num=int((seg[1]-seg[0])/cadence*up_factor)+1, endpoint=True)
        up_time.extend(seg_time)
        up_flux.extend(inter_flux(seg_time))
        bad_time_start = seg[1]

    return np.array(up_time), np.array(up_flux)

def create_bins(lower_bound, upper_bound, num_bins, overlap_factor):
    width = (upper_bound - lower_bound)/num_bins
    bins = []
    for low in np.arange(lower_bound, upper_bound, width)[:num_bins]:
        bins.append([low, low+width])

    adjust = (overlap_factor - 1) / 2.0 * width
    for b in bins:
        b[0] -= adjust
        b[1] += adjust
        
    return bins


def lightcurve_detrending(time, flux, period, epoch, duration):

    transits = np.arange(epoch, time[-1], period)
    lc_detrend = flatten(time,
        		  flux,
        		  method='median',
        		  window_length=1.0,
        		  edge_cutoff=0.0,
        		  break_tolerance=0.5,
        		  cval=5)
    #print(lc_detrend)
    '''
    plt.subplot(211)
    #plt.title('{} {} '.format(ticid, disposition))
    plt.plot(time, flux, '.')
    plt.plot(transits,np.nanmin(flux)*np.ones(len(transits))+1.0,'r^')
    plt.subplot(212)
    plt.plot(time, lc_detrend, '.')
    plt.plot(transits,np.nanmin(lc_detrend)*np.ones(len(transits)),'r^')
    plt.show()
    '''
    return time, lc_detrend

def fill_empty_bin(y): ## indention adjusted
    i = 0
    while i < len(y):
        if np.isnan(y[i]):
            left = i-1
            right = i+1
            # Find nearest non-NaN values on both sides
            while left >= 0 and np.isnan(y[left]):
                left -= 1
            while right < len(y) and np.isnan(y[right]):
                right += 1
            if left >= 0 and right < len(y):
                slope = (y[right] - y[left]) / (right - left)
                for j in range(left + 1, right):
                    y[j] = y[left] + slope*(j - left)
            elif left < 0 and right < len(y):
                y[:right] = y[right]
            elif left >= 0 and right == len(y):
                y[left+1:] = y[left]
            else:
                raise ValueError('Light curve consists only of invalid values')
        i += 1
    return y


def binning(folded_time, flux, bins, normalize=False):
    data = np.zeros((3, len(bins)))
    for i, b in enumerate(bins):
        in_bin = np.logical_and(folded_time > b[0], folded_time < b[1]) ## indices of flux within the bin
        flux_in_bin = flux[in_bin]
        #print(flux_in_bin)
        try:
            data[0,i] = np.quantile(flux_in_bin, 0.25)
        #print(data[0,i])
        except(IndexError):
            data[0,i] = np.nan
        try:
            data[1,i] = np.quantile(flux_in_bin, 0.5)
        except(IndexError):
            data[1,i] = np.nan
        try:
            data[2,i] = np.quantile(flux_in_bin, 0.75)
        except(IndexError):
            data[2,i] = np.nan
        #xxx = input()


    for i in [1, 0, 2]:
        data[i] = fill_empty_bin(data[i])
        if normalize:
            median_flux = np.median(data[1])
            #min_flux = np.min(data[1])
            data[i] -= np.median(data[i])#median_flux
            data[i] /= np.abs(np.min(data[1]))#min_flux)
    return data

csv_filename = '../target_info/adjusted_tces.csv'
#lightcuvre_dir = '../data/lc_data/qlp/qlp_tess_llc/'
lightcuvre_dir = '../fits_files/'

## read information for all tces 
tces = pd.read_csv(csv_filename)

dispositions = tces['Disposition']
tic_ids = tces['tic_id']
sectors = tces['Sectors']
epochs = tces['Epoc']
periods = tces['Period']
duration = tces['Duration']
splits = tces['set']
dupliate = tces['Dupliate']
target_pixel = tces['Target_Pixel']

''' 
## read the patch data
patch_data = np.load('../data/lc_patch_data.npz', allow_pickle=True)
print(patch_data.files)
patch_rowid = patch_data['row']
patch_tics = patch_data['tic']
patch_time = patch_data['time']
patch_flux = patch_data['flux']
patch_quality = patch_data['quality']

print(len(patch_tics))
print(len(set(patch_tics)))

## prepare a dictionary to get flux and quality from tics
patch_time_dict = dict()
patch_flux_dict = dict()
patch_quality_dict = dict()

for tic, time, flux, quality in zip(patch_tics, patch_time, patch_flux, patch_quality):
    patch_time_dict[tic] = time
    patch_flux_dict[tic] = flux
    patch_quality_dict[tic] = quality
'''

i=0
max_points = 0
min_points = 1000000

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []

tics_train, tics_val, tics_test = [], [], []
global_train, local_train, secondary_train, dispositions_train, psd_train = [], [], [], [], []
global_val, local_val, secondary_val, dispositions_val, psd_val = [], [], [], [], []
global_test, local_test, secondary_test, dispositions_test, psd_test = [], [], [], [], []
stellar_params_train, stellar_params_val, stellar_params_test = [], [], []
cadence = 30.0/24.0/60.0 ## QLP 30min cadence in days
seg_limit = 5.0*cadence

### inspect the spoc data 
''' 
duration_list = []
tot, shortdu = 0, 0
for d, t, s, p, e, du, sp, dup, tp in zip(dispositions, tic_ids, sectors, periods, epochs, duration, splits, dupliate, target_pixel):
    if sp in ('train', 'val', 'test') and d in ('EB', 'PC', 'O', 'IS', 'J', 'V') and dup == 'no' and tp == 'yes':
        duration_list.append(du)
        if du<8:
            shortdu += 1
        tot += 1
print(shortdu, tot, shortdu/float(tot))
        
plt.hist(duration_list,bins=50)
plt.show()
plt.close()

xxx = input()'''

start_time = tm.time()
n_global, n_local = 201, 61
n_half_global = int((n_global - 1)/2)
n_half_local = int((n_local - 1)/2)

tot = 0
cnt = 0
for d, t, s, p, ep, du, sp, dup, tp in zip(dispositions, tic_ids, sectors, periods, epochs, duration, splits, dupliate, target_pixel):
    #tot += 1
    #if tot>300:
    #    break
    if sp in ('train', 'val', 'test') and d in ('EB', 'PC', 'O', 'IS', 'J', 'V') and dup == 'no' and tp == 'yes':# and tot>2000:
        hdulist = fits.open(lightcuvre_dir + 'sector{:}/{:016d}.fits'.format(int(s), int(t)))
        quality = hdulist[1].data['QUALITY']
        time = hdulist[1].data['time']
        #flux = hdulist[1].data['KSPSAP_FLUX']
        flux = hdulist[1].data['PDCSAP_FLUX']

        ep = np.mod(ep - time[0], p) + time[0]
        ## flag the nan flux point quality as "bad ones"    
        nan_flag = np.logical_and(np.isnan(flux), np.isnan(time))
        #print(nan_flag) 
        quality[nan_flag] = 1024

        good_points = quality < 1 ## quality==0 are good points
        num_points = np.sum(good_points)

        ## excluded the bad quality and infinite points
        time = time[good_points]
        flux = flux[good_points]

        folded_time = (time - ep + 0.5*p) % p
        phase = np.floor((time - ep + 0.5*p) / p)

        ## normalize step one: set median flux in each phase to be one
        n_phase = set(phase)
        for ph in n_phase:
            phase_median = np.nanmedian(flux[phase==ph])
            flux[phase==ph] /= phase_median

        #flux_median = np.nanmedian(flux)
        #flux /= flux_median
            
        ## global views and local views with the numpy digitize method
        #global_bins = np.linspace(0, p, n_global) ## global view bins        
        #digitized = np.digitize(folded_time, global_bins)
        #global_view = [np.nanmedian(flux[digitized == i]) for i in range(1, n_global, 1)]
        #n_dots_in_bins = [len(flux[digitized == i]) for i in range(1, n_global, 1)]

        #local_bins = np.linspace(0.5*p-du/24, 0.5*p+du/24, n_local) ## global view bins
        #digitized = np.digitize(folded_time, local_bins)
        #local_view = [np.nanmedian(flux[digitized == i]) for i in range(1, n_local, 1)]

        ## global views and local views of quantiles within bins

        global_bins = create_bins(0, p, n_global, 1.0) ## global view bins
        large_global_bins = create_bins(0, p, n_global, 20.0) ## large global view is used to locate the secondary transit
        global_view = []
        large_global_view = []
        for b, lb in zip(global_bins, large_global_bins):
            flux_in_bin = flux[np.logical_and(folded_time > b[0], folded_time < b[1])]
            flux_in_large_bin = flux[np.logical_and(folded_time > lb[0], folded_time < lb[1])]
            ## To address the empty bin issue, expand the bin size until the bin is not empty
            if len(flux_in_bin) == 0:
                expand = p/n_global
                while True:
                    b[0] -= expand
                    b[1] += expand
                    flux_in_bin = flux[np.logical_and(folded_time > b[0], folded_time < b[1])]
                    if len(flux_in_bin)>0:
                        break
                    
            if len(flux_in_large_bin) == 0:
                expand = p/n_global*20
                while True:
                    lb[0] -= expand
                    lb[1] += expand
                    flux_in_large_bin = flux[np.logical_and(folded_time > lb[0], folded_time < lb[1])]
                    if len(flux_in_large_bin)>0:
                        break

            global_view.append(np.nanquantile(flux_in_bin, [0.25, 0.5, 0.75]))
            large_global_view.append(np.nanmedian(flux_in_large_bin))


        local_bins = create_bins(0.5*p-du/16, 0.5*p+du/16, n_local, 1.0) ## local view bins, range 3 times duration
        local_view = []
        for b in local_bins:
            flux_in_bin = flux[np.logical_and(folded_time > b[0], folded_time < b[1])]
            if len(flux_in_bin) == 0:
                expand = p/n_local
                while True:
                    b[0] -= expand
                    b[1] += expand
                    flux_in_bin = flux[np.logical_and(folded_time > b[0], folded_time < b[1])]
                    if len(flux_in_bin)>0:
                        break
            local_view.append(np.nanquantile(flux_in_bin, [0.25, 0.5, 0.75]))

        '''
        ## ## find the time of the the secondary view
        ## padded the folded time and flux, in order to eleminate the "edge effect"
        folded_time = np.concatenate((folded_time-p, folded_time, folded_time+p))
        flux = np.concatenate((flux, flux, flux))
        phase = np.concatenate((phase, phase, phase))

        ## the bin size in the secondary view should match the one in the local view, but bin range in one period
        n_half_secondary = int(np.floor(p/(3*du/24/61)/2))
        n_secondary = int(n_half_secondary*2 + 1) ## better to be an odd number

        ## the difference of flux in the smaller bin and larger bin is used to determine the center of the secondary view
        secondary_small_bins = create_bins(0, p, n_secondary, 1.0)
        secondary_large_bins = create_bins(0, p, n_secondary, 30.0) 

        secondary_small_view, secondary_large_view = [], []
        for sb, lb in zip(secondary_small_bins, secondary_large_bins):
            flux_in_small_bin = flux[np.logical_and(folded_time > sb[0], folded_time < sb[1])]
            flux_in_large_bin = flux[np.logical_and(folded_time > lb[0], folded_time < lb[1])]
            ## To address the empty bin issue, expand the bin size
            if len(flux_in_small_bin) == 0:
                #empty_bin_flag = True
                expand = p/n_secondary
                while True:
                    sb[0] -= expand
                    sb[1] += expand
                    flux_in_small_bin = flux[np.logical_and(folded_time > sb[0], folded_time < sb[1])]
                    if len(flux_in_small_bin)>0:
                        break
            if len(flux_in_large_bin) == 0:
                #empty_bin_flag = True
                expand = p/n_secondary
                while True:
                    lb[0] -= expand
                    lb[1] += expand
                    flux_in_large_bin = flux[np.logical_and(folded_time > lb[0], folded_time < lb[1])]
                    if len(flux_in_large_bin)>0:
                        break

            secondary_small_view.append(np.nanmean(flux_in_small_bin))
            secondary_large_view.append(np.nanmean(flux_in_large_bin))'''
        ## the largest one indicating the most possible transit
        secondary_transit_flag = np.array(large_global_view) - np.array(global_view)[:,1]
        ## mask out the positions already in the local view
        n_half_mask = int(du/p*12.356557377) ## 3*du/24/61*30/(p/201) --- the range in the global view already in the local view
        secondary_transit_flag[n_half_global-n_half_mask:n_half_global+n_half_mask+1] = -1000.
        ## find the most possible secondary tranist time (in bins)
        position_secondary_transit= np.argmax(secondary_transit_flag)

        secondary_transit_time = np.mean(global_bins[position_secondary_transit])

        primary_secondary_diff = abs(secondary_transit_time - 0.5*p) / p ## might be a useful feature
        #print(primary_secondary_diff)
        #secondary_bins = create_bins(secondary_transit_time-du/16, secondary_transit_time+du/16, n_local, 1.0) ## local view bins
        secondary_bins = local_bins ##create_bins(0.5*p-du/16, 0.5*p+du/16, n_local, 1.0) ## local view bins        
        secondary_view = []
        ## padd the flux time series so that the secondary
        ## shift the folded time to put the secondary transit in the middle
        folded_time = (time - ep + secondary_transit_time) % p
        #folded_time = np.concatenate((folded_time-p, folded_time, folded_time+p))
        #flux = np.concatenate((flux, flux, flux))
        #phase = np.concatenate((phase, phase, phase))

        for b in secondary_bins:
            ## NOTICE: folded_time flux have been padded
            flux_in_bin = flux[np.logical_and(folded_time > b[0], folded_time < b[1])]
            if len(flux_in_bin) == 0:
                expand = p/n_local
                while True:
                    b[0] -= expand
                    b[1] += expand
                    flux_in_bin = flux[np.logical_and(folded_time > b[0], folded_time < b[1])]
                    if len(flux_in_bin)>0:
                        break
            secondary_view.append(np.nanquantile(flux_in_bin, [0.25, 0.5, 0.75]))

        

        #secondary_view = np.concatenate((secondary_view[-n_half_local:], secondary_view, secondary_view[:n_half_local]))


        ## normalization step 2: set the minimun of the global view and local view to be zero
        global_view = np.array(global_view)
        local_view = np.array(local_view)
        secondary_view = np.array(secondary_view)

        median_min = np.min(global_view[:,1])
        global_view -= median_min
        global_view /= (1-median_min)

        median_min = np.min(local_view[:,1])
        local_view -= median_min
        local_view /= (1-median_min)

        secondary_flux_median = np.nanmedian(secondary_view[:,1])
        secondary_view /= secondary_flux_median

        median_min = np.min(secondary_view[:,1])
        secondary_view -= median_min
        secondary_view /= ((1-median_min)+1e-7)

        
        #rint(global_view.shape, local_view.shape)

        
        ## draw some figures to see if the data processing is correct.
        if d in ('EB', 'PC'):
            figure, axis = plt.subplots(5, 1)
            axis[0].set_title('TIC: '+str(t)+' Disposition: '+d)
            #axis[0].plot(time, flux, 'k.')
            axis[1].scatter(folded_time, flux, c=phase, cmap='rainbow')
            axis[2].plot(global_view[:,0], 'c.')
            axis[2].plot(global_view[:,1], 'm.')
            axis[2].plot(global_view[:,2], 'g.')
            axis[3].plot(local_view[:,0], 'c.')
            axis[3].plot(local_view[:,1], 'm.')
            axis[3].plot(local_view[:,2], 'g.')
            axis[4].plot(secondary_view[:,0], 'c.')
            axis[4].plot(secondary_view[:,1], 'm.')
            axis[4].plot(secondary_view[:,2], 'g.')
            #axis[4].plot(secondary_small_view, 'b.')
            #axis[4].plot(secondary_large_view, 'r.')
            #plt.plot(time, flux, 'c.')
            #plt.plot(global_view, 'c.')
            plt.show()
            plt.close()

        if sp == 'train':
            tics_train.append(t)
            global_train.append(global_view)
            local_train.append(local_view)
            secondary_train.append(secondary_view)
            dispositions_train.append(d)
            psd_train.append(primary_secondary_diff)

        if sp == 'val':
            tics_val.append(t)
            global_val.append(global_view)
            local_val.append(local_view)
            secondary_val.append(secondary_view)
            dispositions_val.append(d)
            psd_val.append(primary_secondary_diff)
            
        if sp == 'test':
            tics_test.append(t)
            global_test.append(global_view)
            local_test.append(local_view)
            secondary_test.append(secondary_view)
            dispositions_test.append(d)
            psd_test.append(primary_secondary_diff)
            
        cnt += 1
        if cnt%100==0:
            print(cnt, ' time: ', tm.time()-start_time)
            

#pz = input('press enter to save npz files')
## convert the data into numpy format and swap axis to fit the pytorch requirement
global_train = np.array(global_train)
local_train = np.array(local_train)
secondary_train = np.array(secondary_train)

global_train = np.swapaxes(global_train, 1, 2)
local_train = np.swapaxes(local_train, 1, 2)
secondary_train = np.swapaxes(secondary_train, 1, 2)


global_val = np.array(global_val)
local_val = np.array(local_val)
secondary_val = np.array(secondary_val)

global_val = np.swapaxes(global_val, 1, 2)
local_val = np.swapaxes(local_val, 1, 2)
secondary_val = np.swapaxes(secondary_val, 1, 2)


global_test = np.array(global_test)
local_test = np.array(local_test)
secondary_test = np.array(secondary_test)

global_test = np.swapaxes(global_test, 1, 2)
local_test = np.swapaxes(local_test, 1, 2)
secondary_test = np.swapaxes(secondary_test, 1, 2)

print(global_train.shape, local_train.shape, global_val.shape, local_val.shape, global_test.shape, local_test.shape)

np.savez('../model_input/data_q/train.npz', tic=np.array(tics_train), psd=np.array(psd_train),
         global_view=global_train, local_view=local_train, secondary_view=secondary_train, dispositions=np.array(dispositions_train))
np.savez('../model_input/data_q/val.npz', tic=np.array(tics_val), psd=np.array(psd_val),
         global_view=global_val, local_view=local_val, secondary_view=secondary_val, dispositions=np.array(dispositions_val))
np.savez('../model_input/data_q/test.npz', tic=np.array(tics_test), psd=np.array(psd_test),
         global_view=global_test, local_view=local_test, secondary_view=secondary_test, dispositions=np.array(dispositions_test))

print("Preprocessing Finished!")

