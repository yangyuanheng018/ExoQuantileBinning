'''
read the fits files of a sector
pre-process with binning by quantiles
save as a .npz file for the network for infer
'''
import argparse

import pandas as pd
import numpy as np
import os
from astropy.io import fits
from sklearn import preprocessing
import matplotlib.pyplot as plt

import time as tm

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


parser = argparse.ArgumentParser()
parser.add_argument('sector', help='sector', type=str)
args = parser.parse_args()

#lightcuvre_dir = '/media/rluo/Elements/astro_data/TESS_TCE_lc_fits/sector'+args.sector+'/'
#lightcuvre_dir = '../data/sector'+args.sector+'_tce_fits/'  ## for sector 35-43
lightcuvre_dir = '../fits_files/sector'+args.sector+'/' ## for sector 1-5

tce_folder = '../tess_lc_download_sh/tess_tce_csv/'

tce_info_csv_files = os.listdir(tce_folder)
tce_info_csv_files = np.sort(tce_info_csv_files)


sector_stat = pd.read_csv(tce_folder+tce_info_csv_files[int(args.sector)-1], header=6) ## tce_info_csv[0] coresponding to sector 1, etc. 
#sector28 = pd.read_csv('tess_tce_csv/tess2020213081515-s0028-s0028_dvr-tcestats.csv', header=6)

'''
if int(args.sector)<=12:
    ticid = sector_stat['ticid']
    tceid = sector_stat['tceid']
    periods = sector_stat['orbitalPeriodDays']
    durations = sector_stat['transitDurationHours']
    epochs = sector_stat['transitEpochBtjd']
    srad = sector_stat['starRadiusSolarRadii']
    teff = sector_stat['starTeffKelvin']
    logg = sector_stat['starLoggCgs']

else:
    ticid = sector_stat['ticid']
    tceid = sector_stat['tceid']
    periods = sector_stat['tce_period']
    durations = sector_stat['tce_duration']
    epochs = sector_stat['tce_time0bt']
    srad = sector_stat['tce_sradius']
    teff = sector_stat['tce_steff']
    logg = sector_stat['tce_slogg']'''


ticid = sector_stat['ticid']
tceid = sector_stat['tceid']
periods = sector_stat['tce_period']
durations = sector_stat['tce_duration']
epochs = sector_stat['tce_time0bt']
srad = sector_stat['tce_sradius']
teff = sector_stat['tce_steff']
logg = sector_stat['tce_slogg']

    
#nigra = pd.read_csv('tess_tce_csv/nigraha_sec6.csv')


fits_files = os.listdir(lightcuvre_dir)

## extract tics from fits files
fits_tics = []
if int(args.sector)<=9:
    for f in fits_files:
        fits_tics.append(int(f.split('.')[0]))
else:
    fits_filename_header = fits_files[0][:24]
    fits_filename_tail = fits_files[0][40:]
    for f in fits_files:
        fits_tics.append(int(f[24:40]))

start_time = tm.time()

n_global, n_local = 201, 61
n_half_global = int((n_global - 1)/2)
n_half_local = int((n_local - 1)/2)

sun_rad = 432376.0 # (miles)
jupiter_rad = 43441.0 # (miles)


tics_sector, tces_sector, global_sector, local_sector, secondary_sector, scalar_sector = [], [], [], [], [], []
cnt = 0
for t, tce, p, du, ep, sr, tf, lg in zip(ticid, tceid, periods, durations, epochs, srad, teff, logg):
    if t in fits_tics:
        #print(p, du)
        #if t == 42991745:
        #hdulist = fits.open(lightcuvre_dir + 'sector{:}/{:016d}.fits'.format(int(s), int(t)))
        #print(t,end=',')
        if int(args.sector)<=9:
            hdulist = fits.open(lightcuvre_dir + '{:016d}.fits'.format(int(t))) ## the TCEs fits file name
        else:
            hdulist = fits.open(lightcuvre_dir + fits_filename_header+'{:016d}'.format(int(t))+fits_filename_tail) ## the TCEs fits file name

        quality = hdulist[1].data['QUALITY']
        time = hdulist[1].data['time']
        flux = hdulist[1].data['PDCSAP_FLUX'] ## alternatively, use ['KSPSAP_FLUX']

        '''
        if t == 55139557:
            plt.plot(time, flux, 'b.')
            plt.show()
            plt.close()
        
            plt.plot(time, mom_1, 'b.')
            plt.show()
            plt.close()
            plt.plot(time, mom_2, 'b.')
            plt.show()
            plt.close()
            print(t,' data loaded.')'''

        ep = np.mod(ep - time[0], p) + time[0]
        ## flag the nan flux point quality as "bad ones"    
        nan_flag = np.logical_or(np.isnan(flux), np.isnan(time))
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

        ################### GLOBAL VIEW #####################
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

        ################### LOCAL VIEW #####################
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

        ################### SECONDARY VIEW #####################
        ## the largest one indicating the most possible transit
        secondary_transit_flag = np.array(large_global_view) - np.array(global_view)[:,1]
        ## mask out the positions already in the local view
        n_half_mask = int(du/p*12.356557377) ## 3*du/24/61*30/(p/201) --- the range in the global view already in the local view
        secondary_transit_flag[n_half_global-n_half_mask:n_half_global+n_half_mask+1] = -1000.
        ## find the most possible secondary tranist time (in bins)
        position_secondary_transit= np.argmax(secondary_transit_flag)

        secondary_transit_time = np.mean(global_bins[position_secondary_transit])

        #primary_secondary_diff = abs(secondary_transit_time - 0.5*p) / p ## might be a useful feature
        #secondary_bins = create_bins(secondary_transit_time-du/16, secondary_transit_time+du/16, n_local, 1.0) ## local view bins
        secondary_bins = local_bins ##create_bins(0.5*p-du/16, 0.5*p+du/16, n_local, 1.0) ## local view bins        
        secondary_view = []
        secondary_mom_1, secondary_mom_2 = [], []
        ## padd the flux time series so that the secondary
        ## shift the folded time to put the secondary transit in the middle
        folded_shift_time = (time - ep + secondary_transit_time) % p
        folded_time_secondary_centered = (time - ep + secondary_transit_time) % p             
        #folded_time = np.concatenate((folded_time-p, folded_time, folded_time+p))
        #flux = np.concatenate((flux, flux, flux))
        #phase = np.concatenate((phase, phase, phase))

        for b in secondary_bins:
            ## NOTICE: folded_time flux have been padded
            flux_in_bin = flux[np.logical_and(folded_time_secondary_centered > b[0], folded_time_secondary_centered < b[1])]

            if len(flux_in_bin) == 0:
                expand = p/n_local
                while True:
                    b[0] -= expand
                    b[1] += expand
                    flux_in_bin = flux[np.logical_and(folded_time_secondary_centered > b[0], folded_time_secondary_centered < b[1])]

                    if len(flux_in_bin)>0:
                        break
            secondary_view.append(np.nanquantile(flux_in_bin, [0.25, 0.5, 0.75]))

        #secondary_view = np.concatenate((secondary_view[-n_half_local:], secondary_view, secondary_view[:n_half_local]))


        ## normalization step 2: set the minimun of the global view and local view to be zero
        global_view = np.array(global_view)

        local_view = np.array(local_view)

        secondary_view = np.array(secondary_view)

        ## rescale global view
        median_min = np.nanmin(global_view[:,1])
        global_view -= median_min
        global_view /= (1-median_min+1e-7)

        ## CALCULATE PLANET RADIUS
        transit_depth = 1-median_min ## an auxiliary feature
        planet_radius = sr * np.sqrt(transit_depth) * sun_rad / jupiter_rad

        ## rescale local view
        median_min = np.nanmin(local_view[:,1])
        local_view -= median_min
        local_view /= (1-median_min+1e-7)

        ## rescale secondary view
        secondary_flux_median = np.nanmedian(secondary_view[:,1])
        secondary_view /= secondary_flux_median

        median_min = np.nanmin(secondary_view[:,1])
        secondary_view -= median_min
        secondary_view /= ((1-median_min)+1e-7)

        global_view = np.nan_to_num(global_view, nan=1)
        local_view = np.nan_to_num(local_view, nan=1)
        secondary_view = np.nan_to_num(secondary_view, nan=1)


        ## concatenate the flux local view and centroid local views
        #global_view = np.concatenate((global_view, global_mom_1, global_mom_2), axis=1)
        #local_view = np.concatenate((local_view, local_mom_1, local_mom_2), axis=1)
        #secondary_view = np.concatenate((secondary_view, secondary_mom_1, secondary_mom_2), axis=1)

        #rint(global_view.shape, local_view.shape)

        '''
        ## draw some figures to see if the data processing is correct.
        figure, axis = plt.subplots(5, 1)
        axis[0].set_title('TIC: '+str(t))
        axis[0].plot(time, flux, 'k.')
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
        plt.close()'''

        tics_sector.append(t)
        tces_sector.append(tce)
        global_sector.append(global_view)
        local_sector.append(local_view)
        secondary_sector.append(secondary_view)
        scalar_sector.append([p, du, tf, sr, lg, planet_radius])
        cnt += 1
        if cnt%100==0:
            print(cnt, ' time: ', tm.time()-start_time)
    else:
        print('missing fits file, tic: ', t)    
            

#pz = input('press enter to save npz files')
## convert the data into numpy format and swap axis to fit the pytorch requirement
global_view = np.array(global_sector)
local_view = np.array(local_sector)
secondary_view = np.array(secondary_sector)
scalar = np.array(scalar_sector)

global_view = np.swapaxes(global_view, 1, 2)
local_view = np.swapaxes(local_view, 1, 2)
secondary_view = np.swapaxes(secondary_view, 1, 2)

print(global_view.shape, local_view.shape, secondary_view.shape)

np.savez('../model_input/data_q/sector'+str(args.sector)+'.npz', tic=np.array(tics_sector), tce=np.array(tces_sector), \
         global_view=global_view, local_view=local_view, secondary_view=secondary_view, \
         scalar=scalar)


print("Sector "+args.sector+" Preprocessing Finished!")

            
'''
figure, axis = plt.subplots(2, 1)
axis[0].set_title('TIC: '+str(t))
axis[0].plot(time, flux, 'm.')
axis[1].scatter(folded_time, flux, c=phase, cmap='rainbow')
plt.show()
plt.close()

print(t, p, du, ep)'''






'''
sector_stat = open('tess_tce_csv/tess2018349182739-s0006-s0006_dvr-tcestats.csv', 'r')
data = sector_stat.readlines()
sector_stat.close()

for l in data:
    print(l)
    pz = input()'''
