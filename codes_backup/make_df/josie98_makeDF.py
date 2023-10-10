import pandas as pd
import numpy as np
import glob
from pathlib import Path
import re
from homogenization_functions import pumptemp_corr
from analyse_functions import VecInterpolate_log
from constant_variables import *

# Read the metadata file
dfmeta = pd.read_excel("/home/poyraden/Analysis/JOSIEfiles/JOSIE-96-02/Josie_1998_metadata_51added.xls")

for i in range(len(dfmeta)):
    if(dfmeta.at[i,'SondeTypeNr'] < 2): dfmeta.at[i,'ENSCI'] = 0
    if(dfmeta.at[i,'SondeTypeNr'] == 2): dfmeta.at[i,'ENSCI'] = 1
    if(dfmeta.at[i,'SST_Nr'] == 1):
        dfmeta.at[i,'Sol'] = 1.0
        dfmeta.at[i,'Buf'] = 1.0
    if(dfmeta.at[i,'SST_Nr'] == 2):
        dfmeta.at[i,'Sol'] = 0.5
        dfmeta.at[i,'Buf'] = 0.5
    if(dfmeta.at[i,'SST_Nr'] == 3):
        dfmeta.at[i,'Sol'] = 2.0
        dfmeta.at[i,'Buf'] = 0.0
    if(dfmeta.at[i,'SST_Nr'] == 4):
        dfmeta.at[i,'Sol'] = 1.0
        dfmeta.at[i,'Buf'] = 0.1
    if(dfmeta.at[i,'SST_Nr'] == 5):
        dfmeta.at[i,'Sol'] = 2.0
        dfmeta.at[i,'Buf'] = 0.1

# all files
## use pathlib jspr
filenames = dfmeta.Data_FIleName.tolist()
path = '/home/poyraden/Analysis/JOSIEfiles/JOSIE-96-02/1998/JS98-DS0/'
filenames = [path + i for i in filenames]
filenamespath = [Path(j) for j in filenames]

list_data = []

#Some declarations

columnMeta  = ['JOSIE_Nr', 'Sim_Nr', 'R1_Tstart', 'R1_Tstop', 'R2_Tstart', 'R2_Tstop', 'GAW_Report_Nr_Details',
               'Part_Nr' , 'SondeTypeNr', 'SST_Nr', 'Data_FIleName', 'ENSCI', 'Sol', 'Buf']

columnString = "Rec_Nr Time_Day Time_Sim Pres_ESC Temp_ESC Temp_Inlet Alt_Sim PO3_OPM I_ECC_RAW Temp_ECC Cur_Motor " \
               "PO3_ECC_RAW PO3_ECC_BG1 PO3_ECC_BG2 PO3_ECC_BG3 PO3_ECC_BG4 Validity_Nr"
columnStr = columnString.split(" ")


#**********************************************
# Main loop to merge all data sets
#**********************************************
for filename in filenamespath:
    file = open(filename, 'r', encoding="ISO-8859-1")
    print(filename)
    line1 = file.readline()
    matchObj = re.search(r"JS98SN(.*)\.DS0 ", line1)
    team = int(matchObj.group(1)[0:2])
    print('team', team)
    infolist = file.readlines()[0:47]
    print('infolist[9]', infolist[9].split("*")[0])
    # print('infolist[13]', infolist[13])
    ib0 = float(infolist[9].split("*")[0])
    ib1 = float(infolist[13].split("*")[0])
    print(ib0, ib1)

    sim = int(infolist[3].split("*")[0])
    PFcor = float(infolist[10].split("*")[0])/60 * 0.975
    print('sim', sim)

    df = pd.read_csv(filename, engine="python", sep="\s+", skiprows=53, names=columnStr)
    #     ,  encoding = "ISO-8859-1"

    #     # Add the header information to the main df
    df = df.join(pd.DataFrame(
        [[sim, team,  PFcor,ib0, ib1]],
        index=df.index,
        columns=['Sim', 'Team', 'PFcor','iB0','iB1']
    ))

    # Get the index of the metadata that corresponds to this Simulation Number and Participant (Team)

    select_indicesTeam = list(np.where(dfmeta["Part_Nr"] == df['Team'][0]))[0]
    select_indicesSim = list(np.where(dfmeta["Sim_Nr"] == df['Sim'][0]))[0]

    common = [i for i in select_indicesTeam if i in select_indicesSim]
    index_common = common[0]

    list_md = dfmeta.iloc[index_common, :].tolist()

    ## Add  metadata to the main df
    df = df.join(pd.DataFrame(
        [list_md],
        index=df.index,
        columns=columnMeta
    ))

    ## now convert variables to usual Josie naming conventions

    df['PO3'] = df['PO3_ECC_BG3']
    df['IM'] = df['I_ECC_RAW']
    df['TPint'] = df['Temp_ECC']
    df['Pair'] = df['Pres_ESC']
    df['Tsim'] = df['Time_Sim']
    df['TPext'] = df['TPint']

    #2023 update
    # df['PO3_OPM'] = df['PO3_OPM'] + (df['PO3_OPM'] * 1.29 / 100)
    df['PO3_OPM'] = df['PO3_OPM'] * opm_update

    df['unc_Tpump'] = 0.5
    df['Tpump_cor'], df['unc_Tpump_cor'] = pumptemp_corr(df, 'case3', 'TPext', 'unc_Tpump',
                                                         'Pair')

    ## convert OPM pressure to current
    df['I_OPM'] = (df['PO3_OPM'] * df['PFcor']) / (df['TPint'] * 0.043085)
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

    size = len(df)
    Ums_i = [0] * size
    Ua_i = [0] * size
    Ums_i[0] = df.at[0, 'IM']


    ## only convolute slow part of the signal, which is needed for beta calculation
    for i in range(size-1):

        Ua_i = df.at[i + 1, 'I_OPM_jma']
        t1 = df.at[i + 1,'Tsim']
        t2 = df.at[i,'Tsim']
        Xs = np.exp(-(t1 - t2) / slow)
        Ums_i[i + 1] = Ua_i - (Ua_i - Ums_i[i]) * Xs

    df['I_conv_slow'] = Ums_i

    list_data.append(df)
#

    #  end of the allfiles loop    #
     
# Merging all the data files to df

df = pd.concat(list_data,ignore_index=True)


print(list(df))

df.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie1998_Data_allcolumns.csv")

    
df = df.drop(df[((df.Validity_Nr == 0))].index)

df = df.drop(['Rec_Nr', 'Time_Day', 'Time_Sim', 'Pres_ESC', 'Temp_ESC', 'Temp_Inlet', 'Alt_Sim', 'I_ECC_RAW','Temp_ECC',
              'Cur_Motor', 'PO3_ECC_RAW', 'PO3_ECC_BG1', 'PO3_ECC_BG2', 'PO3_ECC_BG3', 'PO3_ECC_BG4', 'Validity_Nr',
              'Sim_Nr', 'GAW_Report_Nr_Details', 'Part_Nr','Data_FIleName'], axis=1)


clist =['JOSIE_Nr','Tsim', 'Sim', 'Team', 'ENSCI', 'Sol', 'Buf','iB0', 'iB1', 'Pair','PO3','PO3_dqa', 'IM','TPint', 'PO3_OPM',
       'I_conv_slow', 'I_OPM_kom', 'I_OPM_jma', 'Tpump_cor', 'unc_Tpump_cor','PFcor', 'R1_Tstart', 'R1_Tstop',
        'R2_Tstart', 'R2_Tstop', 'SST_Nr', 'SondeTypeNr','Cpf_jma','Cpf_kom','PFcor_jma', 'PFcor_kom','unc_Cpf_jma', 'unc_Cpf_kom']
df = df.reindex(columns=clist)

# df.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie1998_Data_updjma_51added.csv")
df.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie1998_Data_2023.csv")

print('new',list(df))



