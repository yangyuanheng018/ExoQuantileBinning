import numpy as np
import pandas as pd

tce_files = ['tess2018349182739-s0006-s0006_dvr-tcestats.csv',
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

output_file = open('appendix_table.txt', 'w+')
output_file.write('##tce_id&tic_id&sectors&tce_period&tce_epoch&tce_duration&tce_depth&tce_steff&tce_slogg&tce_sradius&probability&tfopwg_disp\n')

toi_info_file = pd.read_csv('../target_info/TOI_2021.08.27_23.15.10.csv', header=71)
sectors_dict = dict()
for index in range(len(tce_files)):
    tce_prediction_file = pd.read_csv(f'../tce_probability/tce_exopplanet_probability_sector{6+index}.csv', header=0)
    tce_info_file = pd.read_csv('./tess_tce_csv/' + tce_files[index], header=6)
    sector = 6+index    
    for tce_pred in tce_prediction_file.iterrows():
        # print(tce_pred[1]['probability'], tce_info[1]['orbitalPeriodDays'])
        if tce_pred[1]['probability'] > 0.0:
            ticid = tce_pred[1]['ticid']
            tceid = tce_pred[1]['tceid']
            tce_info = tce_info_file[tce_info_file['tceid']==tceid]
            try:
                tce_id = tce_info['tceid'].values[0]
                tce_period = tce_info['tce_period'].values[0]
                tce_epoch = tce_info['tce_time0bt'].values[0]
                tce_duration = tce_info['tce_duration'].values[0]
                tce_depth = tce_info['tce_depth'].values[0]
                tce_steff = tce_info['tce_steff'].values[0]
                tce_slogg = tce_info['tce_slogg'].values[0]
                tce_sradius = tce_info['tce_sradius'].values[0]
            except:
                tce_id = tce_info['tceid'].values[0]
                tce_period = tce_info['orbitalPeriodDays'].values[0]
                tce_epoch = tce_info['transitEpochBtjd'].values[0]#-2457000
                tce_duration = tce_info['transitDurationHours'].values[0]
                tce_depth = tce_info['transitDepthPpm'].values[0]
                tce_steff = tce_info['starTeffKelvin'].values[0]
                tce_slogg = tce_info['starLoggCgs'].values[0]
                tce_sradius = tce_info['starRadiusSolarRadii'].values[0]
            #print(tce_id)
            #tce_epoch = tce_info[1]['tce_time0bt']
            # print((ticid==toi_info_file['tid']).any())
            # xxx = input()
            tce_key = f"{ticid}&{tce_period:.3f}&{tce_epoch:.3f}&{tce_duration:.3f}&{int(tce_depth)}"
            if tce_key in sectors_dict.keys():
                sectors_dict[tce_key] = sector#.append(sector)
            else:
                sectors_dict[tce_key] = sector

for index in range(len(tce_files)):
    tce_prediction_file = pd.read_csv(f'../tce_probability/tce_exopplanet_probability_sector{6+index}.csv', header=0)
    tce_info_file = pd.read_csv('./tess_tce_csv/' + tce_files[index], header=6)
    sector = 6+index    
    for tce_pred in tce_prediction_file.iterrows():
        # print(tce_pred[1]['probability'], tce_info[1]['orbitalPeriodDays'])
        if tce_pred[1]['probability'] > 0.0:
            ticid = tce_pred[1]['ticid']
            tceid = tce_pred[1]['tceid']
            tce_info = tce_info_file[tce_info_file['tceid']==tceid]
            #print(ticid, sector, tce_info)
            try:
                tce_id = tce_info['tceid'].values[0]
                tce_period = tce_info['tce_period'].values[0]
                tce_epoch = tce_info['tce_time0bt'].values[0]
                tce_duration = tce_info['tce_duration'].values[0]
                tce_depth = tce_info['tce_depth'].values[0]
                tce_steff = tce_info['tce_steff'].values[0]
                tce_slogg = tce_info['tce_slogg'].values[0]
                tce_sradius = tce_info['tce_sradius'].values[0]
            except:
                tce_id = tce_info['tceid'].values[0]
                tce_period = tce_info['orbitalPeriodDays'].values[0]
                tce_epoch = tce_info['transitEpochBtjd'].values[0]#-2457000
                tce_duration = tce_info['transitDurationHours'].values[0]
                tce_depth = tce_info['transitDepthPpm'].values[0]
                tce_steff = tce_info['starTeffKelvin'].values[0]
                tce_slogg = tce_info['starLoggCgs'].values[0]
                tce_sradius = tce_info['starRadiusSolarRadii'].values[0]
            #print(tce_id)
            #tce_epoch = tce_info['tce_time0bt']
            # print((ticid==toi_info_file['tid']).any())
            # xxx = input()
            tce_key = f"{ticid}&{tce_period:.3f}&{tce_epoch:.3f}&{tce_duration:.3f}&{int(tce_depth)}"
            if (ticid == toi_info_file['tid']).any() and (np.around(tce_period, 2) == np.around(
                    toi_info_file['pl_orbper'][ticid == toi_info_file['tid']].values, 2)).any():
                tfopwg_disp = toi_info_file['tfopwg_disp'][ticid == toi_info_file['tid']].values[
                    np.around(tce_period, 2) == np.around(
                        toi_info_file['pl_orbper'][ticid == toi_info_file['tid']].values, 2)]
                # print(ticid, f"{tce_pred[1]['probability']:.3f}", tfopwg_disp[0])  # tce_period, toi_info_file['pl_orbper'][ticid==toi_info_file['tid']].values, tce_epoch+2457000, toi_info_file['pl_tranmid'][ticid==toi_info_file['tid']].values
                output_file.write(
                    f"{tce_id}&{ticid}&{sectors_dict[tce_key]}&{tce_period:.3f}&{tce_epoch:.3f}&{tce_duration:.3f}&{int(tce_depth)}&{int(tce_steff)}&{tce_slogg:.3f}&{tce_sradius:.3f}&{tce_pred[1]['probability']:.3f}&{tfopwg_disp[0]}\\\\\n")
            else:
                # print(ticid, f"{tce_pred[1]['probability']:.3f}")
                # xxx = input()
                output_file.write(
                    f"{tce_id}&{ticid}&{sectors_dict[tce_key]}&{tce_period:.3f}&{tce_epoch:.3f}&{tce_duration:.3f}&{int(tce_depth)}&{int(tce_steff)}&{tce_slogg:.3f}&{tce_sradius:.3f}&{tce_pred[1]['probability']:.3f}&00\\\\\n")

output_file.close()

df = pd.read_csv('appendix_table.txt', sep='&', header=0)
print(df[:10])
df.sort_values(by='probability', ascending=False, inplace=True)
print(df[:10])
df.drop_duplicates(subset=['tic_id', 'tce_period', 'tce_epoch', 'tce_duration', 'tce_depth'], keep='first', inplace=True)
print(df[:10])
df.to_csv('appendix_table.txt', sep='&', index=False)

df = pd.read_csv('appendix_table.txt', sep='&', header=0)
df.drop(index=df[df['tfopwg_disp']!='00\\\\'].index, inplace=True)
df.drop(columns='tfopwg_disp',inplace=True)
print(df[:100])
df.to_csv('appendix_table_clean.txt', sep='&', index=False)

