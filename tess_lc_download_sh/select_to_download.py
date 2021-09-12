'''
generate the download script of TCE TICs in a sector
'''
import argparse

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('sector', help='sector', type=int)
args = parser.parse_args()

index = args.sector - 6

## read all curl command lines from the shell file
'''curl_files = ['tesscurl_lc_sh/tesscurl_sector_35_lc.sh',
              'tesscurl_lc_sh/tesscurl_sector_36_lc.sh',
              'tesscurl_lc_sh/tesscurl_sector_37_lc.sh',
              'tesscurl_lc_sh/tesscurl_sector_38_lc.sh',
              'tesscurl_lc_sh/tesscurl_sector_39_lc.sh',]'''
curl_shell = open(f'tesscurl_lc_sh/tesscurl_sector_{args.sector}_lc.sh')
commands = curl_shell.readlines()
curl_shell.close()

## read the tce tic ids from the csv file
tce_files = ['tess2018349182737-s0006-s0006_dvr-tcestats.csv',
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

tce_csv = pd.read_csv('./tess_tce_csv/'+tce_files[index], header=6)
tce_ticids = set(tce_csv['ticid'])

print(len(tce_ticids))
#print(tce_ticid)

## if the tic in the curl command is also in the tce_ticids, select the command and write it into a new file
out_file = open('tesscurl_sector_'+str(args.sector)+'_tce_lc.sh', 'w')
out_file.write(commands[0])
cnt = 0 
for l in commands[1:]:
    ticid = int(l.split('-')[6])
    if ticid in tce_ticids:
        out_file.write(l)
        cnt += 1

print(cnt)
out_file.close()
