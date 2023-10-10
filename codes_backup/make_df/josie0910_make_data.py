import pandas as pd
import numpy as np
import re
import glob
import math
from math import log
from analyse_functions import polyfit, VecInterpolate_log
from homogenization_functions import pumptemp_corr
from constant_variables import *

##########  part for Pump Temperature Cell:
df17 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_deconv.csv", low_memory=False)

p_en, p_sp = polyfit(df17)

#######################################################################
#constant variables

# year = '2009'
year = '2010'

# Read the metadata and data files
dfmeta = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Josie{year[2:4]}_MetaData.csv")
dfmeta_ib = pd.read_excel(f"/home/poyraden/Analysis/JOSIEfiles/ib_{year}.xlsx")
dfmeta_m = pd.read_excel(f"/home/poyraden/Analysis/JOSIEfiles/{year}_mass.xlsx")
allFiles = glob.glob(f"/home/poyraden/Analysis/JOSIEfiles/JOSIE-{year}-Data-ReProc/*.O3R")



list_data = []

# Some declarations
columnString = "Tact Tsim Pair Tair Tinlet IM TPint TPext PFcor I_Pump PO3 VMRO3 PO3_OPM VMRO3_OPM ADif_PO3S RDif_PO3S Z"
columnStr = columnString.split(" ")
columnMeta = ['Year', 'Sim', 'Team', 'Code', 'Flow', 'IB1', 'Cor', 'ENSCI', 'Sol', 'Buf', 'ADX']
columnMeta_ib = ['Simib', 'Teamib', 'Yearib', 'iB0', 'iB1', 'iB2']
columnMeta_m = ['Simm', 'Teamm', 'Mspre', 'Mspost', 'Diff']


# **********************************************
# Main loop to merge all data sets
# **********************************************
for filename in allFiles:
    file = open(filename, 'r')
    # Get the participant information from the name of the file
    print(filename)
    # Get the participant information from the name of the file
    tmp_team = filename.split("/")[6]  # 7 for home, 6 for kmi
    header_team = (tmp_team.split("_")[2]).split(".")[0]
    file.readline()
    file.readline()
    header_part = int(header_team)
    header_sim = int(file.readline().split()[2])
    file.readline()
    file.readline()
    header_PFunc = float(file.readline().split()[1])
    header_PFcor = float(file.readline().split()[1])
    print(header_sim, header_part, header_PFcor)
    file.readline()
    header_IB1 = float(file.readline().split()[1])
    print(filename, 'header_IB1', header_IB1)

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

    ## now the same for ib0 values
    select_indicesTeam_ib = list(np.where(dfmeta_ib["Team"] == df['Header_Team'][0]))[0]
    select_indicesSim_ib = list(np.where(dfmeta_ib["Sim"] == df['Header_Sim'][0]))[0]

    common_ib = [i for i in select_indicesTeam_ib if i in select_indicesSim_ib]
    index_common_ib = common_ib[0]

    ## now the same for mass diff values
    select_indicesTeam_m = list(np.where(dfmeta_m["Team"] == df['Header_Team'][0]))[0]
    select_indicesSim_m = list(np.where(dfmeta_m["Sim"] == df['Header_Sim'][0]))[0]

    common_m = [i for i in select_indicesTeam_m if i in select_indicesSim_m]
    index_common_m = common_m[0]

    ## The index of the metadata that has the information of this simulation = index_common
    #  assign this row into a list
    list_md = dfmeta.iloc[index_common, :].tolist()
    list_md_ib = dfmeta_ib.iloc[index_common_ib, :].tolist()
    list_md_m = dfmeta_m.iloc[index_common_m, :].tolist()

    ## Add  metadata to the main df
    df = df.join(pd.DataFrame(
        [list_md],
        index=df.index,
        columns=columnMeta
    ))

    df = df.join(pd.DataFrame(
        [list_md_ib],
        index=df.index,
        columns=columnMeta_ib
    ))

    df = df.join(pd.DataFrame(
        [list_md_m],
        index=df.index,
        columns=columnMeta_m
    ))

    # 2023 update:
    df['PO3_OPM'] = df['PO3_OPM'] * opm_update
    ## the TPint values in the data need to be inter-changed with TPext, therefore for calculation of PO3,
    # TPext needs to be used.
    #Truest Pump temperature correction
    # df['TPextC'] = df['TPext'] - kelvin
    df['unc_Tpump'] = 0.5
    df['Tpump_cor'], df['unc_Tpump_cor'] = pumptemp_corr(df, 'case5', 'TPext', 'unc_Tpump', 'Pair')
    
    #PF rate efficiency correctionfilt_en = (dft[j].ENSCI == 1)
    filt_sp = (df.ENSCI == 0)
    filt_en = (df.ENSCI == 1)

    print('begin')

    df.loc[filt_en, 'Cpf_kom'], df.loc[filt_en, 'unc_Cpf_kom'] = VecInterpolate_log(pvallog, komhyr_95, komhyr_95_unc,
                                                                            df[filt_en], 'Pair')
    df.loc[filt_sp, 'Cpf_kom'], df.loc[filt_sp, 'unc_Cpf_kom'] = VecInterpolate_log(pvallog, komhyr_86, komhyr_86_unc,
                                                                            df[filt_sp], 'Pair')
    
    df['Cpf_jma'], df['unc_Cpf_jma'] = VecInterpolate_log(pvallog_jma, JMA, jma_unc, df, 'Pair')
    df['PFcor_kom'] = df['PFcor']/df['Cpf_kom']
    df['PFcor_jma'] = df['PFcor']/df['Cpf_jma']

    print('end')
    ## convert OPM pressure to current
    df['I_OPM_jma'] = (df['PO3_OPM'] * df['PFcor_jma']) / (df['Tpump_cor'] * 0.043085)
    df['I_OPM_kom'] = (df['PO3_OPM'] * df['PFcor_kom']) / (df['Tpump_cor'] * 0.043085)

    df['PO3_calc'] = (0.043085 * df['TPext'] * (df['IM'] - df['iB1'])) / (df['PFcor_kom'])
    df['PO3_dqa'] = (0.043085 * df['Tpump_cor'] * (df['IM'] - df['iB1'])) / (df['PFcor_kom'])


    # for k in range(len(df)):
    #
    #     for pi in range(len(yref_pair) - 1):
    #
    #         if (df.at[k, 'Pair'] >= yref_pair[pi + 1]) & (df.at[k, 'Pair'] < yref_pair[pi]):
    #             if df.at[k, 'ENSCI'] == 1: df.at[k, 'Tcell'] = df.at[k, 'TPext'] - p_en(y_pair[pi])
    #             if df.at[k, 'ENSCI'] == 0: df.at[k, 'Tcell'] = df.at[k, 'TPext'] - p_sp(y_pair[pi])
    #
    #     if (df.at[k, 'Pair'] <= yref_pair[19]):
    #         if df.at[k, 'ENSCI'] == 1: df.at[k, 'Tcell'] = df.at[k, 'TPext'] - p_en(y_pair[18])
    #         if df.at[k, 'ENSCI'] == 0: df.at[k, 'Tcell'] = df.at[k, 'TPext'] - p_sp(y_pair[18])
    #
    #     df.at[k, 'TcellC'] = df.at[k, 'Tcell'] - 273
    #
    #     for t in range(len(Temp) - 1):
    #
    #         if (df.at[k, 'TcellC'] >= Temp[t]) & (df.at[k, 'TcellC'] < Temp[t + 1]):
    #             df.at[k, 'Pw'] = Pw_temp[t]
    #
    #     if (df.at[k, 'TcellC'] >= Temp[10]): df.at[k, 'Pw'] = Pw_temp[10]
    #     if (df.at[k, 'TcellC'] < Temp[0]): df.at[k, 'Pw'] = Pw_temp[0]
    #
    #
    #     df.at[k, 'massloss'] = 0.0001 * 1000 * (df.at[k, 'Pw'] / (461.52 * df.at[k, 'Tcell'])) * df.at[k, 'JMA'] * \
    #                            df.at[k, 'PFcor']
    #     df.at[k, 'Tboil'] = ((237.3 * np.log10(df.at[k, 'Pair'])) - 186.47034) / (8.2858 - np.log10(df.at[k, 'Pair']))
    #
    # df['total_massloss'] = np.trapz(df.massloss, x=df.Tsim)

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

    df.loc[filt2, 'Pair'] = np.array(df.loc[filt1, 'Pair'])
    df.loc[filt3, 'Pair'] = np.array(df.loc[filt1, 'Pair'])
    df.loc[filt4, 'Pair'] = np.array(df.loc[filt1, 'Pair'])


# correct Pair values, and assign them to the first participants Pair values

df.to_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_Data_2023paper.csv")

# 2023 -> OPM updated to * 1.29 and I_OPM_JMA added
# paper -> I_OPM is calculated using TPump corrected and Pf_cor_pumpflow_efficiency
