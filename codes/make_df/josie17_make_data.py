#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import re
import glob
import math
from math import log
from analyse_functions import polyfit, VecInterpolate_log
from homogenization_functions import pumptemp_corr
from constant_variables import *

##########  part for TP Cell:

df17 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_deconv.csv", low_memory=False)

p_en, p_sp = polyfit(df17)

print('p_ensci:', np.array(p_en))
print('p_spc:', np.array(p_sp))

#######################################################################


# Read the metadata file
dfmeta = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/JOSIE_Table_2017_summary.csv")

# Path to all Josie17 simulation files
allFiles = glob.glob("/home/poyraden/Analysis/JOSIEfiles/Josie17/*.O3R")
list_data = []

# Some declarations
columnString = "Tact Tsim Pair Tair Tinlet IM TPint TPext PFcor I_Pump PO3 VMRO3 PO3_OPM VMRO3_OPM ADif_PO3S RDif_PO3S Z"
columnStr = columnString.split(" ")

# related with metadata
column_metaString = 'Sim Simulator_RunNr Date Team Ini_Prep_Date Prep_SOP SerialNr ENSCI SerNr Date_1 SondeAge ' \
                    'Solutions Sol Buf Volume_Cathode ByPass_Cell Current_10min_after_noO3 Resp_Time_4_1p5_sec' \
                    ' RespTime_1minOver2min Final_BG iB0 iB1 iB2 T100 mlOvermin T100_post mlOverminp1 ' \
                    'RespTime_4_1p5_sec_p1 RespTime_1minOver2min_microamps PostTestSolution_Lost_gr PumpMotorCurrent ' \
                    'PumpMotorCurrent_Post PF_Unc PF_Cor BG'
columnMeta = column_metaString.split(" ")


# **********************************************
# Main loop to merge all data sets
# **********************************************
for filename in allFiles:
    file = open(filename, 'r')
    # Get the participant information from the name of the file
    print(filename)
    tmp_team = filename.split("/")[6]
    header_team = (tmp_team.split("_")[2]).split(".")[0]

    # Not a very nice way of getting the headers, but it works :)
    # Be careful for different file formats, then needs to be changed
    file.readline()
    file.readline()
    header_part = float(header_team)
    header_sim = int(file.readline().split()[2])
    file.readline()
    file.readline()
    header_PFunc = float(file.readline().split()[1])
    header_PFcor = float(file.readline().split()[1])
    file.readline()
    header_IB1 = float(file.readline().split()[1])

    # Assign the main df
    df = pd.read_csv(filename, sep="\t", engine="python", skiprows=12, names=columnStr)

    # Add the header information to the main df
    df = df.join(pd.DataFrame(
        [[header_part, header_sim, header_PFunc, header_PFcor, header_IB1]],
        index=df.index,
        columns=['Header_Team', 'Header_Sim', 'Header_PFunc', 'Header_PFcor', 'Header_IB1']
    ))

    # Get the index of the metadata that corresponds to this Simulation Number and Participant (Team)
    select_indicesTeam = list(np.where(dfmeta["Team"] == df['Header_Team'][0]))[0]
    select_indicesSim = list(np.where(dfmeta["Sim"] == df['Header_Sim'][0]))[0]
    common = [i for i in select_indicesTeam if i in select_indicesSim]
    index_common = common[0]

    ## The index of the metadata that has the information of this simulation = index_common
    #  assign this row into a list
    list_md = dfmeta.iloc[index_common, :].tolist()

    ## Add  metadata to the main df
    df = df.join(pd.DataFrame(
        [list_md],
        index=df.index,
        columns=columnMeta
    ))


    # 2023 update:
    df['PO3_OPM'] = df['PO3_OPM'] * opm_update
    ## the TPint values in the data need to be inter-changed with TPext, therefore for calculation of PO3,
    # TPext needs to be used.
    # Truest Pump temperature correction
    # df['TPextC'] = df['TPext'] - kelvin
    df['unc_Tpump'] = 0.5
    df['Tpump_cor'], df['unc_Tpump_cor'] = pumptemp_corr(df, 'case5', 'TPext', 'unc_Tpump', 'Pair')

    # PF rate efficiency correctionfilt_en = (dft[j].ENSCI == 1)
    filt_sp = (df.ENSCI == 0)
    filt_en = (df.ENSCI == 1)
    filt_01 = (df.Buf == 0.1)
    not_filt_01 = (df.Buf != 0.1)


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

    df.loc[filt_01, 'PO3_calc'] = (0.043085 * df.loc[filt_01,'TPext'] * (df.loc[filt_01, 'IM'] - df.loc[filt_01,'iB2']))\
                                  / (df.loc[filt_01, 'PFcor_jma'])
    df.loc[not_filt_01, 'PO3_calc'] = (0.043085 * df.loc[not_filt_01,'TPext'] * (df.loc[not_filt_01, 'IM'] - df.loc[not_filt_01,'iB2']))\
                                  / (df.loc[not_filt_01, 'PFcor_kom'])

    df.loc[filt_01, 'PO3_dqa'] = (0.043085 * df.loc[filt_01,'Tpump_cor'] * (df.loc[filt_01, 'IM'] - df.loc[filt_01,'iB2']))\
                                  / (df.loc[filt_01, 'PFcor_jma'])
    df.loc[not_filt_01, 'PO3_dqa'] = (0.043085 * df.loc[not_filt_01,'Tpump_cor'] * (df.loc[not_filt_01, 'IM'] - df.loc[not_filt_01,'iB2']))\
                                  / (df.loc[not_filt_01, 'PFcor_kom'])


    for k in range(len(df)):
        ## jma corrections for OPM current, OPM_I_jma will be used only for Ua in the convolution of
        ## the slow component of the signal
        for pi in range(len(yref_pair) -1):

            if (df.at[k, 'Pair'] >= yref_pair[pi + 1]) & (df.at[k, 'Pair'] < yref_pair[pi]) :
                if df.at[k, 'ENSCI'] == 1: df.at[k, 'Tcell'] = df.at[k, 'TPext'] - p_en(y_pair[pi])
                if df.at[k, 'ENSCI'] == 0: df.at[k, 'Tcell'] = df.at[k, 'TPext'] - p_sp(y_pair[pi])

        if df.at[k, 'Pair'] <= yref_pair[19]:
            if df.at[k, 'ENSCI'] == 1: df.at[k, 'Tcell'] = df.at[k, 'TPext'] - p_en(y_pair[18])
            if df.at[k, 'ENSCI'] == 0: df.at[k, 'Tcell'] = df.at[k, 'TPext'] - p_sp(y_pair[18])

        df.at[k, 'TcellC'] = df.at[k, 'Tcell'] - 273

        for t in range(len(Temp) - 1):

            if (df.at[k, 'TcellC'] >= Temp[t]) & (df.at[k, 'TcellC'] < Temp[t + 1]):
                df.at[k, 'Pw'] = Pw_temp[t]

        if df.at[k, 'TcellC'] >= Temp[10]: df.at[k, 'Pw'] = Pw_temp[10]
        if df.at[k, 'TcellC'] < Temp[0]: df.at[k, 'Pw'] = Pw_temp[0]


        # if (df.at[k, 'Pw'] < df.at[k, 'Pair']):
        df.at[k, 'massloss'] = 0.0001 * 1000 * df.at[k, 'Pw'] / (461.52 * df.at[k, 'Tcell']) *  df.at[k, 'PFcor_jma']
        df.at[k, 'Tboil'] = (( 237.3 * np.log10(df.at[k, 'Pair']) )  - 186.47034)  / (8.2858 - np.log10(df.at[k, 'Pair']))


    df['total_massloss'] = np.trapz(df.massloss, x=df.Tsim)

    size = len(df)
    Ums_i = [0] * size
    Ums_i_kom = [0] * size

    Ua_i = [0] * size
    Ums_i[0] = df.at[0, 'IM']
    Ums_i_kom[0] = df.at[0, 'IM']

    ## only convolute slow part of the signal, which is needed for beta calculation
    for i in range(size - 1):
        Ua_i = df.at[i + 1, 'I_OPM_jma']
        t1 = df.at[i + 1, 'Tsim']
        t2 = df.at[i, 'Tsim']
        Xs = np.exp(-(t1 - t2) / slow)
        Ums_i[i + 1] = Ua_i - (Ua_i - Ums_i[i]) * Xs

        Ua_i = df.at[i + 1, 'I_OPM_kom']
        t1 = df.at[i + 1, 'Tsim']
        t2 = df.at[i, 'Tsim']
        Xs = np.exp(-(t1 - t2) / slow)
        Ums_i_kom[i + 1] = Ua_i - (Ua_i - Ums_i[i]) * Xs

    df['I_conv_slow'] = Ums_i
    df['I_conv_slow_komhyr'] = Ums_i_kom



    list_data.append(df)
    #  end of the allfiles loop    #

# Merging all the data files to df
df = pd.concat(list_data, ignore_index=True)

dfsim = df.drop_duplicates(['Sim'])
simlist = dfsim.Sim.tolist()

for s in simlist:
    filt1 = (df.Sim == s) & (df.Team == 1)
    filt2 = (df.Sim == s) & (df.Team == 2)
    filt3 = (df.Sim == s) & (df.Team == 3)
    filt4 = (df.Sim == s) & (df.Team == 4)
    filt5 = (df.Sim == s) & (df.Team == 5)
    filt6 = (df.Sim == s) & (df.Team == 6)
    filt7 = (df.Sim == s) & (df.Team == 7)
    filt8 = (df.Sim == s) & (df.Team == 8)

    df.loc[filt2, 'Pair'] = np.array(df.loc[filt1, 'Pair'])
    df.loc[filt3, 'Pair'] = np.array(df.loc[filt1, 'Pair'])
    df.loc[filt4, 'Pair'] = np.array(df.loc[filt1, 'Pair'])
    df.loc[filt6, 'Pair'] = np.array(df.loc[filt5, 'Pair'])
    df.loc[filt7, 'Pair'] = np.array(df.loc[filt5, 'Pair'])
    df.loc[filt8, 'Pair'] = np.array(df.loc[filt5, 'Pair'])


# df.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_Data_nocut_2022_updjma.csv")
df.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_Data_2023_paper.csv")
#
# 2023 -> OPM updated to * 1.29 and I_OPM_JMA added
# paper -> I_OPM is calculated using TPump corrected and Pf_cor_pumpflow_efficiency
# In[ ]:


## aply the cuts here for O3 calculation
#
# df = df.drop(df[((df.Sim == 171) | (df.Sim == 172) | (df.Sim == 180) | (df.Sim == 185))].index)
# df = df.drop(df[(df.Sim == 179) & (df.Team == 4) & (df.Tsim > 4000)].index)
# df = df.drop(df[(df.Sim == 172) & (df.Tsim < 500)].index)
# df = df.drop(df[(df.Sim == 172) & (df.Team == 1) & (df.Tsim > 5000) & (df.Tsim < 5800)].index)
# df = df.drop(df[(df.Sim == 178) & (df.Team == 3) & (df.Tsim > 1700) & (df.Tsim < 2100)].index)
# df = df.drop(df[(df.Sim == 178) & (df.Team == 3) & (df.Tsim > 2500) & (df.Tsim < 3000)].index)
#
# df = df.drop(df[((df.Sim == 175))].index)
# df = df.drop(df[((df.Tsim > 7000))].index)

