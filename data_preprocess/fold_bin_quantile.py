from astropy.io import fits
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal
from sklearn import preprocessing
#from wotan import flatten
import time as tm
import matplotlib.pyplot as plt

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

csv_filename = '../target_info/modified_disposition_subgrouped_tces_2.csv'
#lightcuvre_dir = '../data/lc_data/qlp/qlp_tess_llc/'
lightcuvre_dir = '../fits_files/'

tces = pd.read_csv(csv_filename)
#print(tces.columns)

dispositions = tces['second_modifications']
tic_ids = tces['tic_id']
sectors = tces['Sectors']
epochs = tces['Epoc']
periods = tces['Period']
duration = tces['Duration']
srad = tces['star_rad']
transit_depth = tces['Transit_Depth']
#splits = tces['set']
#dupliate = tces['Dupliate']
target_pixel = tces['Target_Pixel']
subgroups = tces['subgroup'] ## for training-validation-test split

##

stellar = np.load('../data/stellar.npy')
## Teff, sradius, logg, mass (NOT standardized) 
#print(stellar.shape)
#print(subgroups.shape)

## check if the imputed stellar parameters are aligned --- YES they are aligned
#for i in range(40,10000,401):
#    print(srad[i], stellar[i,1])
#ps = input()


cadence = 30.0/24.0/60.0 ## QLP 30min cadence in days
seg_limit = 5.0*cadence
start_time = tm.time()
n_global, n_local = 201, 61
n_half_global = int((n_global - 1)/2)
n_half_local = int((n_local - 1)/2)

sun_rad = 432376.0 # (miles)
jupiter_rad = 43441.0 # (miles)


tot = 0
cnt = 0
local_view_list = []
global_view_list = []
secondary_view_list = []
scalar_list = []
subgroup_list = []
disposition_list = []
start_time = tm.time()
for d, t, s, p, ep, du, g, tp, st in zip(dispositions, tic_ids, sectors, periods, epochs, duration, subgroups, target_pixel, stellar):
    tot += 1
    if g>-0.5 and tp == 'yes': ## 2-min cadence data only
        #start_time = tm.time()

        hdulist = fits.open(lightcuvre_dir + 'sector{:}/{:016d}.fits'.format(int(s), int(t)))
        quality = hdulist[1].data['QUALITY']
        time = hdulist[1].data['time']
        #flux = hdulist[1].data['KSPSAP_FLUX']
        flux = hdulist[1].data['PDCSAP_FLUX']
        mom_1 = hdulist[1].data['MOM_CENTR1']
        mom_2 = hdulist[1].data['MOM_CENTR2']

        ep = np.mod(ep - time[0], p) + time[0]
        ## flag the nan flux point quality as "bad ones"    
        nan_flag = np.logical_and(np.isnan(flux), np.isnan(time))
        #print(nan_flag) 
        quality[nan_flag] = 1024

        nan_flag = np.logical_or(np.isnan(mom_1), np.isnan(mom_2))

        ## in case of (almost) empty mom data, fill with random numbers
        if np.sum(np.logical_not(nan_flag))<200: 
            mom_1 = np.random.randn(len(time))
            mom_2 = np.random.randn(len(time))
            nan_flag = np.logical_and(np.isnan(mom_1), np.isnan(mom_2))

        quality[nan_flag] = 1024

        good_points = quality < 1 ## quality==0 are good points
        num_points = np.sum(good_points)

        ## excluded the bad quality and infinite points
        time = time[good_points]
        flux = flux[good_points]
        mom_1 = mom_1[good_points]
        mom_2 = mom_2[good_points]

        #flux_median = np.median(flux)
        
        folded_time = (time - ep + 0.5*p) % p
        phase = np.floor((time - ep + 0.5*p) / p)

        ## normalize step one: set median flux in each phase to be one
        n_phase = set(phase)
        for ph in n_phase:
            phase_median = np.nanmedian(flux[phase==ph])
            flux[phase==ph] /= phase_median

        #print('load fits file', tm.time()-start_time)
        #start_time = tm.time()

        ################### GLOBAL VIEW #####################
        ## global view bins
        global_bins = create_bins(0, p, n_global, 1.0)
        ## large global view is used to locate the secondary transit
        large_global_bins = create_bins(0, p, n_global, 20.0)
        global_view = []
        global_mom_1, global_mom_2 = [], []
        large_global_view = []
        for b, lb in zip(global_bins, large_global_bins):
            flux_in_bin = flux[np.logical_and(folded_time > b[0], folded_time < b[1])]
            mom_1_in_bin = mom_1[np.logical_and(folded_time > b[0], folded_time < b[1])]
            mom_2_in_bin = mom_2[np.logical_and(folded_time > b[0], folded_time < b[1])]

            flux_in_large_bin = flux[np.logical_and(folded_time > lb[0], folded_time < lb[1])]
            ## To address the empty bin issue, expand the bin size until the bin is not empty
            if len(flux_in_bin) == 0:
                expand = p/n_global
                while True:
                    b[0] -= expand
                    b[1] += expand
                    flux_in_bin = flux[np.logical_and(folded_time > b[0], folded_time < b[1])]
                    mom_1_in_bin = mom_1[np.logical_and(folded_time > b[0], folded_time < b[1])]
                    mom_2_in_bin = mom_2[np.logical_and(folded_time > b[0], folded_time < b[1])]
                    
                    if len(flux_in_bin)>0:
                        break
            ## large global view bins

            if len(flux_in_large_bin) == 0:
                expand = p/n_global*20
                while True:
                    lb[0] -= expand
                    lb[1] += expand
                    flux_in_large_bin = flux[np.logical_and(folded_time > lb[0], folded_time < lb[1])]
                    if len(flux_in_large_bin)>0:
                        break

            global_view.append(np.nanquantile(flux_in_bin, [0.25, 0.5, 0.75]))
            global_mom_1.append([np.nanmedian(mom_1_in_bin)])
            global_mom_2.append([np.nanmedian(mom_2_in_bin)])
            large_global_view.append(np.nanmedian(flux_in_large_bin))

        ################### LOCAL VIEW #####################
        ## local view bins, range 3 times duration
        local_bins = create_bins(0.5*p-du/16, 0.5*p+du/16, n_local, 1.0)
        local_view = []
        local_mom_1, local_mom_2 = [], []
        for b in local_bins:
            flux_in_bin = flux[np.logical_and(folded_time > b[0], folded_time < b[1])]
            mom_1_in_bin = mom_1[np.logical_and(folded_time > b[0], folded_time < b[1])]
            mom_2_in_bin = mom_2[np.logical_and(folded_time > b[0], folded_time < b[1])]
            if len(flux_in_bin) == 0:
                expand = p/n_local
                while True:
                    b[0] -= expand
                    b[1] += expand
                    flux_in_bin = flux[np.logical_and(folded_time > b[0], folded_time < b[1])]
                    mom_1_in_bin = mom_1[np.logical_and(folded_time > b[0], folded_time < b[1])]
                    mom_2_in_bin = mom_2[np.logical_and(folded_time > b[0], folded_time < b[1])]
                    if len(flux_in_bin)>0:
                        break
            local_view.append(np.nanquantile(flux_in_bin, [0.25, 0.5, 0.75]))
            #local_mom_1.append(np.nanquantile(mom_1_in_bin, [0.25, 0.5, 0.75]))
            #local_mom_2.append(np.nanquantile(mom_2_in_bin, [0.25, 0.5, 0.75]))
            local_mom_1.append([np.nanmedian(mom_1_in_bin)])
            local_mom_2.append([np.nanmedian(mom_2_in_bin)])

        ################### SECONDARY VIEW #####################
        ## the largest one of the flag indicating the most possible transit
        secondary_transit_flag = np.array(large_global_view) - np.array(global_view)[:,1]
        ## mask out the positions already in the local view
        n_half_mask = int(du/p*12.356557377) ## 3*du/24/61*30/(p/201) --- the range in the global view already in the local view
        secondary_transit_flag[n_half_global-n_half_mask:n_half_global+n_half_mask+1] = -1000.
        ## find the most possible secondary tranist time (in bins)
        position_secondary_transit= np.argmax(secondary_transit_flag)

        secondary_transit_time = np.mean(global_bins[position_secondary_transit])

        #primary_secondary_diff = abs(secondary_transit_time - 0.5*p) / p ## might be a useful feature
        #print(primary_secondary_diff)
        #secondary_bins = create_bins(secondary_transit_time-du/16, secondary_transit_time+du/16, n_local, 1.0) ## local view bins
        secondary_bins = local_bins ##create_bins(0.5*p-du/16, 0.5*p+du/16, n_local, 1.0) ## local view bins        
        secondary_view = []
        secondary_mom_1, secondary_mom_2 = [], []
        ## padd the flux time series so that the secondary
        ## shift the folded time to put the secondary transit in the middle
        folded_time_secondary_centered = (time - ep + secondary_transit_time) % p
        #folded_time = np.concatenate((folded_time-p, folded_time, folded_time+p))
        #flux = np.concatenate((flux, flux, flux))
        #phase = np.concatenate((phase, phase, phase))



        for b in secondary_bins:
            ## NOTICE: folded_time flux have been padded
            flux_in_bin = flux[np.logical_and(folded_time_secondary_centered > b[0], folded_time_secondary_centered < b[1])]
            mom_1_in_bin = mom_1[np.logical_and(folded_time > b[0], folded_time < b[1])]
            mom_2_in_bin = mom_2[np.logical_and(folded_time > b[0], folded_time < b[1])]

            if len(flux_in_bin) == 0:
                expand = p/n_local
                while True:
                    b[0] -= expand
                    b[1] += expand
                    flux_in_bin = flux[np.logical_and(folded_time_secondary_centered > b[0], folded_time_secondary_centered < b[1])]
                    mom_1_in_bin = mom_1[np.logical_and(folded_time > b[0], folded_time < b[1])]
                    mom_2_in_bin = mom_2[np.logical_and(folded_time > b[0], folded_time < b[1])]

                    if len(flux_in_bin)>0:
                        break
            secondary_view.append(np.nanquantile(flux_in_bin, [0.25, 0.5, 0.75]))
            secondary_mom_1.append([np.nanmedian(mom_1_in_bin)])
            secondary_mom_2.append([np.nanmedian(mom_2_in_bin)])

        ## convert to numpy
        global_view = np.array(global_view)
        global_mom_1 = np.array(global_mom_1)
        global_mom_2 = np.array(global_mom_2)

        local_view = np.array(local_view)
        local_mom_1 = np.array(local_mom_1)
        local_mom_2 = np.array(local_mom_2)
        
        secondary_view = np.array(secondary_view)
        secondary_mom_1 = np.array(secondary_mom_1)
        secondary_mom_2 = np.array(secondary_mom_2)

        ## rescale global view
        median_min = np.nanmin(global_view[:,1])
        global_view -= median_min
        global_view /= (1-median_min)

        ## CALCULATE PLANET RADIUS
        transit_depth = 1-median_min ## an auxiliary feature
        planet_radius = st[1] * np.sqrt(transit_depth) * sun_rad / jupiter_rad
        #f d in ('EB', 'PC'):
        #   print(transit_depth, st[1], planet_radius, d) ## st[1] is the star radius

        ## rescale local view 
        median_min = np.nanmin(local_view[:,1])
        local_view -= median_min
        local_view /= (1-median_min)

        ## rescale secondary view
        secondary_flux_median = np.nanmedian(secondary_view[:,1])
        secondary_view /= secondary_flux_median

        median_min = np.nanmin(secondary_view[:,1])
        secondary_view -= median_min
        secondary_view /= ((1-median_min)+1e-7)

        ## the following may be incorrect
        #median_min = np.min(secondary_view[:,1])
        #secondary_view -= median_min
        #secondary_view /= ((1-median_min)+1e-7)

        ## standardize the centroid views
        scaler = preprocessing.StandardScaler().fit(global_mom_1)
        global_mom_1 = scaler.transform(global_mom_1)
        scaler = preprocessing.StandardScaler().fit(global_mom_2)
        global_mom_2 = scaler.transform(global_mom_2)

        scaler = preprocessing.StandardScaler().fit(local_mom_1)
        local_mom_1 = scaler.transform(local_mom_1)
        scaler = preprocessing.StandardScaler().fit(local_mom_2)
        local_mom_2 = scaler.transform(local_mom_2)

        scaler = preprocessing.StandardScaler().fit(secondary_mom_1)
        secondary_mom_1 = scaler.transform(secondary_mom_1)
        scaler = preprocessing.StandardScaler().fit(secondary_mom_2)
        secondary_mom_2 = scaler.transform(secondary_mom_2)


        ## concatenate the flux local view and centroid local views
        global_view = np.concatenate((global_view, global_mom_1, global_mom_2), axis=1)
        local_view = np.concatenate((local_view, local_mom_1, local_mom_2), axis=1)
        secondary_view = np.concatenate((secondary_view, secondary_mom_1, secondary_mom_2), axis=1)


        '''
        if d in ('EB'):
            figure, axis = plt.subplots(7, 1, figsize= (8,24))
            axis[0].set_title('TIC: '+str(t)+' Disposition: '+d)
            axis[0].plot(time, flux, 'k.')
            #axis[1].plot(folded_time, flux, 'k.')
            axis[1].scatter(folded_time, flux, c=phase, cmap='rainbow')
            axis[2].plot(global_view[:,0], 'c.')
            axis[2].plot(global_view[:,1], 'm.')
            axis[2].plot(global_view[:,2], 'g.')
            #axis[2].plot(global_view[:,3]-3, 'k.')
            #axis[2].plot(global_view[:,4]-3, 'r.')
            axis[3].plot(local_view[:,0], 'c.')
            axis[3].plot(local_view[:,1], 'm.')
            axis[3].plot(local_view[:,2], 'g.')
            axis[4].plot(secondary_view[:,0], 'c.')
            axis[4].plot(secondary_view[:,1], 'm.')
            axis[4].plot(secondary_view[:,2], 'g.')
            axis[5].plot(global_view[:,3], 'r.')
            axis[5].plot(global_view[:,4], 'b.')
            axis[6].plot(local_view[:,3], 'r.')
            axis[6].plot(local_view[:,4], 'b.')
            #axis[4].plot(secondary_small_view, 'b.')
            #axis[4].plot(secondary_large_view, 'r.')
            #plt.plot(time, flux, 'c.')
            #plt.plot(global_view, 'c.')
            plt.show()
            plt.close()'''

        global_view_list.append(global_view)
        local_view_list.append(local_view)
        secondary_view_list.append(secondary_view)

        ## scalar parameters are (in order): period, duration, Teff, radius, logg, planet_radius (NOT standardized):
        scalar_list.append([p,du,st[0],st[1],st[2],planet_radius])
        subgroup_list.append(g)
        if d == 'PC':
            disposition_list.append([0,0,1])
        elif d == 'EB':
            disposition_list.append([0,1,0])
        else:
            disposition_list.append([1,0,0])
        cnt += 1
        if cnt%100==0:
            print(cnt, tm.time()-start_time)

global_view = np.array(global_view_list)
local_view = np.array(local_view_list)
secondary_view = np.array(secondary_view_list)
scalar = np.array(scalar_list)
subgroup = np.array(subgroup_list)
disposition = np.array(disposition_list)

global_view = np.swapaxes(global_view, 1, 2)
local_view = np.swapaxes(local_view, 1, 2)
secondary_view = np.swapaxes(secondary_view, 1, 2)

print(global_view.shape)
print(local_view.shape)
print(secondary_view.shape)
print(scalar.shape)
print(subgroup.shape)
print(disposition.shape)

#print(global_view[:3])
#print(local_view[:3])
#print(secondary_view[:3])
#print(scalar[:3])
#print(subgroup[:3])
#print(disposition[:3])

''' 
for ind in range(100):
    print(disposition[ind])
    if disposition[ind,1]>0.5:
        print(scalar[ind])
        figure, axis = plt.subplots(3, 1, figsize= (8,24))

        axis[0].plot(global_view[ind,:,0], 'c.')
        axis[0].plot(global_view[ind,:,1], 'm.')
        axis[0].plot(global_view[ind,:,2], 'g.')
        axis[0].plot(global_view[ind,:,3], 'k.')
        axis[0].plot(global_view[ind,:,4], 'r.')
        #axis[2].plot(global_view[:,3]-3, 'k.')
        #axis[2].plot(global_view[:,4]-3, 'r.')
        axis[1].plot(local_view[ind,:,0], 'c.')
        axis[1].plot(local_view[ind,:,1], 'm.')
        axis[1].plot(local_view[ind,:,2], 'g.')
        axis[1].plot(local_view[ind,:,3], 'k.')
        axis[1].plot(local_view[ind,:,4], 'r.')
        axis[2].plot(secondary_view[ind,:,0], 'c.')
        axis[2].plot(secondary_view[ind,:,1], 'm.')
        axis[2].plot(secondary_view[ind,:,2], 'g.')
        axis[2].plot(secondary_view[ind,:,3], 'k.')
        axis[2].plot(secondary_view[ind,:,4], 'r.')
        plt.show()
        plt.close()'''


np.savez('../model_input/training_data.npz', global_view=global_view, local_view=local_view, secondary_view=secondary_view, \
         scalar=scalar, subgroup=subgroup, dispositions=dispositions)
