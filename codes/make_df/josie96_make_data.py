import pandas as pd
import numpy as np
import re
import glob
from homogenization_functions import pumptemp_corr
from analyse_functions import VecInterpolate_log
from constant_variables import *


allFiles = glob.glob("/home/poyraden/Analysis/JOSIEfiles/JOSIE-96-02/1996/JOSIE-1996-DS1/DS1-Data/Ascent/*DS1")

columnString = "RecNr Time Tsim Pair Tair Temp_TMC_Ist_1 Altitude PO3_OPM I_Sonde T_Sonde PO3_Sonde_raw " \
               "PO3_Sonde_Corr PO3_Sonde_TCO3 O3_Col_corr O3_Col_OPM Valid"


columnStr = columnString.split(" ")

list_data = []

for filename in allFiles:
    file = open(filename, 'r',  encoding='latin-1')
    # Get the participant information from the name of the file
    # print(filename)
    header_one = (file.readline().split("SIM")[1])
    # print(header_one)
    sim_num = int(header_one[1:3])
    team_num = int(header_one[4:5])
    file.readline()
    header_bkg = (file.readline().split(" =")[0])
    bkg = header_bkg
    bkg = float(header_bkg)
    header_pf = (file.readline().split(" =")[0])
    pf = float(header_pf)
    header_tocs = (file.readline().split(" =")[0])
    toc_s = float(header_tocs)
    header_toco = (file.readline().split(" =")[0])
    toc_o = float(header_toco)

    # if (sim_num != 27 & team_num != 1):continue
    if(sim_num == 27 and team_num == 1) or (sim_num == 33 and team_num == 5) or (sim_num == 33 and team_num == 8):continue

    df = pd.read_csv(filename, sep = "\s+", engine="python", skiprows=8,  names=columnStr,  encoding='latin-1')
    # print(df[0:2])

    df = df.join(pd.DataFrame(
        [[sim_num, team_num,  pf, bkg, toc_s, toc_o]],
        index=df.index,
        columns=[ 'Sim', 'Team', 'PF_min','iB', 'TOCS', 'TOCO']
    ))

    #2023 update
    df['PF'] = df['PF_min']/60. * 0.975
    df['PFcor'] = df['PF']
    df['TPext'] = df['T_Sonde']
    df['IM'] = df['I_Sonde']
    # df['PO3_OPM'] = df['PO3_OPM'] + (df['PO3_OPM'] * 1.29/100)
    df['PO3_OPM'] = df['PO3_OPM'] * opm_update
    df['unc_Tpump'] = 0.5
    df['Tpump_cor'], df['unc_Tpump_cor'] = pumptemp_corr(df, 'case3', 'TPext', 'unc_Tpump',
                                                         'Pair')




    list_data.append(df)
    #  end of the allfiles loop    #

# Merging all the data files to df

df = pd.concat(list_data, ignore_index=True)

df.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie1996_Data_2023_tmp.csv")

ind1 = df[(df['Sim'] < 31) & ((df['Team'] == 1) | (df['Team'] == 2) | (df.Team == 4))].index
df = df.drop(ind1)
ind2 = df[(df['Sim'] > 30) & (df['Sim'] < 37) & (
            (df['Team'] == 5) | (df['Team'] == 7) | (df.Team == 8))].index
df = df.drop(ind2)
df = df[df.Sim != 34]
# df.loc[]
df.loc[df.Sim < 51,'IM'] = df.loc[df.Sim < 51,'I_Sonde']
df.loc[df.Sim < 51,'TPext'] = df.loc[df.Sim < 51,'T_Sonde']
# df.loc[df.Sim > 51,'TPext'] = df.loc[df.Sim > 51,'TPint']

df.loc[(df.Sim < 37) & (df.Team ==3),'Sol'] = 1.0
df.loc[(df.Sim < 37) & (df.Team ==3),'Buf'] = 1.0
df.loc[(df.Sim < 37) & (df.Team ==3),'ENSCI'] = 0

df.loc[(df.Sim < 37) & (df.Team == 6), 'Sol'] = 1.0
df.loc[(df.Sim < 37) & (df.Team == 6), 'Buf'] = 1.0
df.loc[(df.Sim < 37) & (df.Team == 6), 'ENSCI'] = 0


slist_1 = [25, 26, 27, 28, 29, 30]
ib0_1 = [0.01, 0.02, 0.03, 0.01, 0.01, 0.01]
ib1_1 = [0.05, 0.08, 0.11, 0.06, 0.07, 0.05]

slist_2 = [31, 32, 33, 34, 35, 36]
ib0_2 = [0.02, 0.0, 0.02, 0.03, 0.02, 0.01]
ib1_2 = [0.06, 0.05, 0.07, 0.08, 0.06, 0.05]

for k in range(len(slist_1)):
    df.loc[df.Sim == slist_1[k], 'iB0'] = ib0_1[k]
    df.loc[df.Sim == slist_1[k], 'iB1'] = ib1_1[k]

    df.loc[df.Sim == slist_2[k], 'iB0'] = ib0_2[k]
    df.loc[df.Sim == slist_2[k], 'iB1'] = ib1_2[k]

df['ib1-ib0'] = df['iB1'] - df['iB0']
# df = df[df.Sim > 50]

#now asign ensci and bkg values and calculate dqa corrected opm

filt_sp = (df.ENSCI == 0)
filt_en = (df.ENSCI == 1)

print('begin')

df.loc[filt_en, 'Cpf_kom'], df.loc[filt_en, 'unc_Cpf_kom'] = VecInterpolate_log(pvallog, komhyr_95, komhyr_95_unc,
                                                                                df[filt_en], 'Pair')
df.loc[filt_sp, 'Cpf_kom'], df.loc[filt_sp, 'unc_Cpf_kom'] = VecInterpolate_log(pvallog, komhyr_86, komhyr_86_unc,
                                                                                df[filt_sp], 'Pair')

df['Cpf_jma'], df['unc_Cpf_jma'] = VecInterpolate_log(pvallog_jma, JMA, jma_unc, df, 'Pair')
df['PFcor_kom'] = df['PFcor'] / df['Cpf_kom']
df['PFcor_jma'] = df['PFcor'] / df['Cpf_jma']

print('end')
## convert OPM pressure to current
df['I_OPM_jma'] = (df['PO3_OPM'] * df['PFcor_jma']) / (df['Tpump_cor'] * 0.043085)
df['I_OPM_kom'] = (df['PO3_OPM'] * df['PFcor_kom']) / (df['Tpump_cor'] * 0.043085)

df['PO3_calc'] = (0.043085 * df['TPext'] * (df['IM'] - df['iB1'])) / (df['PFcor_kom'])
df['PO3_dqa'] = (0.043085 * df['Tpump_cor'] * (df['IM'] - df['iB1'])) / (df['PFcor_kom'])


df.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie1996_Data_2023.csv")