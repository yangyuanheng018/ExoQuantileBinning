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

#lightcuvre_dir = './sector6_fits_part/'
#lightcuvre_dir = '/media/rluo/Elements/astro_data/TESS_TCE_lc_fits/sector'+args.sector+'/'
lightcuvre_dir = '../fits_files/sector'+args.sector+'_tce_fits/'

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

sector6 = pd.read_csv('../tess_lc_download_sh/tess_tce_csv/'+csv_files[int(args.sector)-6], header=6)
#sector28 = pd.read_csv('tess_tce_csv/tess2020213081515-s0028-s0028_dvr-tcestats.csv', header=6)
if int(args.sector)<=30:
    ticid = sector6['ticid']
    tceid = sector6['tceid']
    periods = sector6['orbitalPeriodDays']
    durations = sector6['transitDurationHours']
    epochs = sector6['transitEpochBtjd']
else:
    ticid = sector6['ticid']
    tceid = sector6['tceid']
    periods = sector6['tce_period']
    durations = sector6['tce_duration']
    epochs = sector6['tce_time0bt']
    

#nigra = pd.read_csv('tess_tce_csv/nigraha_sec6.csv')

fits_files = os.listdir(lightcuvre_dir)

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

tics_sector, global_sector, local_sector, secondary_sector = [], [], [], []
cnt = 0
for t, p, du, ep in zip(ticid, periods, durations, epochs):
    if t in fits_tics:
        #print(p, du)
        #if t == 42991745:
        #hdulist = fits.open(lightcuvre_dir + 'sector{:}/{:016d}.fits'.format(int(s), int(t)))
        if int(args.sector)<=9:
            hdulist = fits.open(lightcuvre_dir + '{:016d}.fits'.format(int(t))) ## the TCEs fits file name
        else:
            hdulist = fits.open(lightcuvre_dir + fits_filename_header+'{:016d}'.format(int(t))+fits_filename_tail) ## the TCEs fits file name

        quality = hdulist[1].data['QUALITY']
        time = hdulist[1].data['time']
        flux = hdulist[1].data['PDCSAP_FLUX'] ## alternatively, use ['KSPSAP_FLUX']

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

        ## the largest one indicating the most possible transit
        secondary_transit_flag = np.array(large_global_view) - np.array(global_view)[:,1]
        ## mask out the positions already in the local view
        n_half_mask = int(du/p*12.356557377) ## 3*du/24/61*30/(p/201) --- the range in the global view already in the local view
        secondary_transit_flag[n_half_global-n_half_mask:n_half_global+n_half_mask+1] = -1000.
        ## find the most possible secondary tranist time (in bins)
        position_secondary_transit= np.argmax(secondary_transit_flag)

        secondary_transit_time = np.mean(global_bins[position_secondary_transit])

        primary_secondary_diff = abs(secondary_transit_time - 0.5*p) / p ## might be a useful feature
        #secondary_bins = create_bins(secondary_transit_time-du/16, secondary_transit_time+du/16, n_local, 1.0) ## local view bins
        secondary_bins = local_bins ##create_bins(0.5*p-du/16, 0.5*p+du/16, n_local, 1.0) ## local view bins        
        secondary_view = []
        ## padd the flux time series so that the secondary
        ## shift the folded time to put the secondary transit in the middle
        folded_shift_time = (time - ep + secondary_transit_time) % p
        #folded_time = np.concatenate((folded_time-p, folded_time, folded_time+p))
        #flux = np.concatenate((flux, flux, flux))
        #phase = np.concatenate((phase, phase, phase))

        for b in secondary_bins:
            ## NOTICE: folded_time flux have been padded
            flux_in_bin = flux[np.logical_and(folded_shift_time > b[0], folded_shift_time < b[1])]
            if len(flux_in_bin) == 0:
                expand = p/n_local
                while True:
                    b[0] -= expand
                    b[1] += expand
                    flux_in_bin = flux[np.logical_and(folded_shift_time > b[0], folded_shift_time < b[1])]
                    if len(flux_in_bin)>0:
                        break
            secondary_view.append(np.nanquantile(flux_in_bin, [0.25, 0.5, 0.75]))

        

        #secondary_view = np.concatenate((secondary_view[-n_half_local:], secondary_view, secondary_view[:n_half_local]))


        ## normalization step 2: set the minimun of the global view and local view to be zero
        global_view = np.array(global_view)
        local_view = np.array(local_view)
        secondary_view = np.array(secondary_view)

        median_min = np.nanmin(global_view[:,1])
        global_view -= median_min
        global_view /= (1-median_min+1e-7)

        median_min = np.nanmin(local_view[:,1])
        local_view -= median_min
        local_view /= (1-median_min+1e-7)

        secondary_flux_median = np.nanmedian(secondary_view[:,1])
        secondary_view /= secondary_flux_median

        median_min = np.nanmin(secondary_view[:,1])
        secondary_view -= median_min
        secondary_view /= ((1-median_min)+1e-7)

        global_view = np.nan_to_num(global_view, nan=1)
        local_view = np.nan_to_num(local_view, nan=1)
        secondary_view = np.nan_to_num(secondary_view, nan=1)

        tics_sector.append(t)
        global_sector.append(global_view)
        local_sector.append(local_view)
        secondary_sector.append(secondary_view)

        cnt += 1
        if cnt%100==0:
            print(cnt, ' time: ', tm.time()-start_time)
    else:
        print('missing fits file, tic: ', t)    
            

#pz = input('press enter to save npz files')
## convert the data into numpy format and swap axis to fit the pytorch requirement
global_sector = np.array(global_sector)
local_sector = np.array(local_sector)
secondary_sector = np.array(secondary_sector)

global_sector = np.swapaxes(global_sector, 1, 2)
local_sector = np.swapaxes(local_sector, 1, 2)
secondary_sector = np.swapaxes(secondary_sector, 1, 2)

print(global_sector.shape, local_sector.shape)

np.savez('../model_input/data_q/sector'+str(args.sector)+'.npz', tic=np.array(tics_sector),
         global_view=global_sector, local_view=local_sector, secondary_view=secondary_sector)

print("Sector "+args.sector+" Preprocessing Finished!")

