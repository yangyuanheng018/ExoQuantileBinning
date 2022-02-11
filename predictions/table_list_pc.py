import os
import numpy as np
import pandas as pd

####################################################
## read statistics for TCEs in sector 6 to 43 sector
####################################################

tce_folder = '../tess_lc_download_sh/tess_tce_csv_new/'

tce_info_csv_files = os.listdir(tce_folder)
tce_info_csv_files = np.sort(tce_info_csv_files)

#print(tce_info_csv_files)
tce_stats_numbers = []
for sector in range(1,44,1):
    #print(sector,tce_folder,tce_info_csv_files[int(sector-1)])
    tce_stats = pd.read_csv(tce_folder+tce_info_csv_files[int(sector-1)], header=6) ## tce_info_csv[0] coresponding to sector 1, etc.
    periods = tce_stats['tce_period']
    epochs = tce_stats['tce_time0bt']
    durations = tce_stats['tce_duration']
    depths = tce_stats['tce_depth']
    teff = tce_stats['tce_steff']
    logg = tce_stats['tce_slogg']
    sradius = tce_stats['tce_sradius']
    tce_stats_numbers.append(len(periods))

    if sector == 1:
        all_periods = periods
        all_epochs = epochs
        all_durations = durations
        all_depths = depths
        all_teff = teff
        all_logg = logg
        all_sradius = sradius
    else:
        all_periods = np.concatenate((all_periods, periods), axis=0)
        all_epochs = np.concatenate((all_epochs, epochs), axis=0)
        all_durations = np.concatenate((all_durations, durations), axis=0)
        all_depths = np.concatenate((all_depths, depths), axis=0)
        all_teff = np.concatenate((all_teff, teff), axis=0)
        all_logg = np.concatenate((all_logg, logg), axis=0)
        all_sradius = np.concatenate((all_sradius, sradius), axis=0)

print(all_periods.shape, all_durations.shape, all_depths.shape, all_teff.shape)

###########################################################
## read EB and PC probabilities for TCEs in sector 6 to 43 
###########################################################

tce_prediction_numbers = []
for sector in range(1,44,1):
    probability = pd.read_csv('./tce_eb_pc_probability_sector_'+str(sector)+'.csv')
    pc_prob = np.array(probability['PC_probability'])
    eb_prob = np.array(probability['EB_probability'])
    tce_ids = np.array(probability['tceid'])
    tic_ids = np.array(probability['ticid'])
    tce_prediction_numbers.append(len(tic_ids))

    #if 459997997 in tic_ids:
    if sector == 1:
        all_pc_prob = pc_prob
        all_eb_prob = eb_prob
        all_tce = tce_ids
        all_tic = tic_ids
        all_sector = np.ones_like(tic_ids) * sector

    else:
        all_pc_prob = np.concatenate((all_pc_prob, pc_prob), axis=0)
        all_eb_prob = np.concatenate((all_eb_prob, eb_prob), axis=0)
        all_tce = np.concatenate((all_tce, tce_ids), axis=0)
        all_tic = np.concatenate((all_tic, tic_ids), axis=0)
        all_sector = np.concatenate((all_sector, np.ones_like(tic_ids) * sector), axis=0)



print(all_pc_prob.shape, all_eb_prob.shape, all_tce.shape, all_tic.shape, all_sector.shape)

for n1, n2 in zip(tce_stats_numbers, tce_prediction_numbers):
    if n1!=n2:
        print('numbers of TCEs in sectors not matched!')
#print(all_tic[[1,1000,5000,10000]])
#print(all_sector[[1,1000,10000,50000]])
#ps = input()

###############
## read TOIs ##
###############


toi = pd.read_csv('../target_info/TOI_2021.12.12_03.47.53.csv', header=69)

tid = toi['tid']
disposition = toi['tfopwg_disp']
period = toi['pl_orbper']

'''print(set(disposition))

cnt = 0
tic_set = set()
for t, d, p  in zip(tid, disposition, period):
    #if d=='CP' or d=='KP':
    if ('F' in str(d) or str(d)=='CP' or str(d)=='KP') and np.isnan(p):
        print(t, p, d)
        tic_set.add(t)
        cnt += 1

print(tic_set)
print(len(tid))'''
####################################################################################
### generate a dictionary to check TFOPWG disposition according to tic and period ##
####################################################################################
ticp_disp_dict = dict()
for t, p, d in zip(tid, period, disposition):
    ticp = str(t)+'|'+str(p)[:3] ## an idendity 
    if ticp not in ticp_disp_dict.keys():
        ticp_disp_dict[ticp] = d
    elif d != ticp_disp_dict[ticp]:
        print(d, ticp_disp_dict[ticp])

#print(ticp_disp_dict)

### sort the all the PC probability
sorted_arg_pc_prob = np.argsort(all_pc_prob)[::-1] ## high PC probability first
#print(np.max(sorted_arg_pc_prob))


#########################################
## table of all top PC probability TCEs #
#########################################
existed_ticp = set()
for t in sorted_arg_pc_prob[:100]:
    ticp = str(all_tic[t])+'|'+str(all_periods[t])[:3]
    #print(t, ticp)
    if ticp in ticp_disp_dict.keys():
        tfopwg_disposition = str(ticp_disp_dict[ticp])
    else:
        tfopwg_disposition = '-'

    #if 'F' in tfopwg_disposition:
    if ticp not in existed_ticp:
        print(str(all_tce[t])+'&' +str(all_tic[t])+'&' + str(all_sector[t])+'&'+str(all_periods[t])[:8]+'&' \
              +str(all_epochs[t])[:8]+'&' + str(all_durations[t])[:8]+'&'+ str(all_depths[t])[:8]+'&'\
              +str(round(all_teff[t]))+'&' + str(all_logg[t])[:6]+'&'+ str(all_sradius[t])[:6]+'&'\
              +str(all_pc_prob[t])[:5]+'&'+tfopwg_disposition+'\\\\')

    existed_ticp.add(ticp)

ps = input('press enter to continue.')
#############################################
## table of top PC probability non-TOI TCEs #
#############################################

existed_ticp = set() ## tag to remove the duplicated TCEs (multi-sector)
for t in sorted_arg_pc_prob[:1000]:
    ticp = str(all_tic[t])+'|'+str(all_periods[t])[:3]
    #print(t, ticp)
    if ticp in ticp_disp_dict.keys():
        tfopwg_disposition = str(ticp_disp_dict[ticp])
    else:
        tfopwg_disposition = '-'

    #if 'F' in tfopwg_disposition:
    if tfopwg_disposition == '-' and ticp not in existed_ticp:
        print(str(all_tce[t])+'&' +str(all_tic[t])+'&' + str(all_sector[t])+'&'+str(all_periods[t])[:8]+'&' \
              +str(all_epochs[t])[:8]+'&' + str(all_durations[t])[:8]+'&'+ str(all_depths[t])[:8]+'&'\
              +str(round(all_teff[t]))+'&' + str(all_logg[t])[:6]+'&'+ str(all_sradius[t])[:6]+'&'\
              +str(all_pc_prob[t])[:5]+'\\\\')

    existed_ticp.add(ticp)


ps = input('press enter to continue.')
###############################################
## table of all PC probability TCEs to a file #
###############################################
table_txt = open('./all_pc_probalility_multi_sector_removed.csv', 'w')
table_txt.write('tce_id,tic_id,sector,period,epoch,duration,depth,Teff,logg,stellar_radius,PC_probability,tfopwg_disposition\n')

existed_ticp = set()
for t in sorted_arg_pc_prob:
    ticp = str(all_tic[t])+'|'+str(all_periods[t])[:3]
    #print(t, ticp)
    if ticp in ticp_disp_dict.keys():
        tfopwg_disposition = ticp_disp_dict[ticp]
    else:
        tfopwg_disposition = '-'

    #if 'F' in tfopwg_disposition:
    #if tfopwg_disposition == '-' and ticp not in existed_ticp:
    #
    #    print(str(all_tce[t])+'&' +str(all_tic[t])+'&' + str(all_sector[t])+'&'+str(all_periods[t])[:8]+'&' \
    #          +str(all_epochs[t])[:8]+'&' + str(all_durations[t])[:8]+'&'+ str(all_depths[t])[:8]+'&'\
    #          +str(round(all_teff[t]))+'&' + str(all_logg[t])[:6]+'&'+ str(all_sradius[t])[:6]+'&'\
    #          +str(all_pc_prob[t])[:5]+'\\\\')
    if ticp not in existed_ticp:
        table_txt.write(str(all_tce[t])+',' +str(all_tic[t])+',' + str(all_sector[t])+','+str(all_periods[t])+',' \
                        +str(all_epochs[t])+',' + str(all_durations[t])+','+ str(all_depths[t])+','\
                        +str(round(all_teff[t]))+',' + str(all_logg[t])+','+ str(all_sradius[t])+','\
                        +str(all_pc_prob[t])+','+str(tfopwg_disposition)+'\n')
    existed_ticp.add(ticp)
                        

table_txt.close()
#results_txt.write(str(all_tic[t])+',' +str(all_tce[t])+',' + str(all_sector[t])+','+str(all_pc_prob[t])+'\n')


