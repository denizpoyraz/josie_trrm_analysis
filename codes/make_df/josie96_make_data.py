import pandas as pd
import numpy as np
import re
import glob
from functions.homogenization_functions import pumptemp_corr


allFiles = glob.glob("/home/poyraden/Analysis/JOSIEfiles/JOSIE-96-02/1996/JOSIE-1996-DS1/DS1-Data/Ascent/*DS1")

columnString = "RecNr Time Tsim Pair Tair Temp_TMC_Ist_1 Altitude PO3_OPM I_Sonde T_Sonde PO3_Sonde_raw " \
               "PO3_Sonde_Corr PO3_Sonde_TCO3 O3_Col_corr O3_Col_OPM Valid"

# Pval = np.array([1020, 730, 535, 382, 267, 185, 126, 85, 58, 39, 26.5, 18.1, 12.1, 8.3, 6])
# JMA = np.array(
#     [0.999705941, 0.997216654, 0.995162562, 0.992733959, 0.989710199, 0.985943645, 0.981029252, 0.974634364,
#      0.966705137, 0.956132227, 0.942864263, 0.9260478, 0.903069813, 0.87528384, 0.84516337])
Pval = [1000,200, 100, 50, 30, 20, 10, 7, 5, 4, 3]
JMA = [1, 0.9881422924901185, 0.9784735812133072, 0.9633911368015414, 0.9478672985781991, 0.929368029739777, 0.8826125330979699, 0.8474576271186441, 0.8071025020177562, 0.7763975155279503, 0.7347538574577517]

#2023 update
Pval_komhyr = np.array([1000, 100, 50, 30, 20, 10, 7, 5, 3])
komhyr_sp_tmp = np.array([1, 1.007, 1.018, 1.022, 1.032, 1.055, 1.070, 1.092, 1.124])  # SP Komhyr 86
komhyr_en_tmp = np.array([1, 1.007, 1.018, 1.029, 1.041, 1.066, 1.087, 1.124, 1.241])  # ENSCI Komhyr 95

komhyr_sp = [1 / i for i in komhyr_sp_tmp]
komhyr_en = [1 / i for i in komhyr_en_tmp]

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
    df['PO3_OPM'] = df['PO3_OPM'] * 1.01293
    df['unc_Tpump'] = 0.5
    df['Tpump_cor'], df['unc_Tpump_cor'] = pumptemp_corr(df, 'case3', 'TPext', 'unc_Tpump',
                                                         'Pair')


    for k in range(len(df)):

        df.at[k, 'PO3_nocorr'] = 0.043085 * df.at[k, 'T_Sonde'] * (df.at[k, 'I_Sonde'] - df.at[k, 'iB']) / df.at[k, 'PF']

        for p in range(len(JMA) - 1):

            if (df.at[k, 'Pair'] >= Pval[p + 1]) & (df.at[k, 'Pair'] < Pval[p]):
                # print(p, Pval[p + 1], Pval[p ])
                df.at[k, 'I_OPM_jma'] = df.at[k, 'PO3_OPM'] * df.at[k, 'PFcor'] * JMA[p] / \
                                        (df.at[k, 'TPext'] * 0.043085)
                df.at[k, 'I_OPM_jma_tpcor'] = df.at[k, 'PO3_OPM'] * df.at[k, 'PFcor'] * JMA[p] / \
                                        (df.at[k, 'Tpump_cor'] * 0.043085)


                df.at[k, 'JMA'] = JMA[p]

        if (df.at[k, 'Pair'] <= Pval[-1]):
            df.at[k, 'I_OPM_jma'] = df.at[k, 'PO3_OPM'] * df.at[k, 'PFcor'] * JMA[-1] / \
                                    (df.at[k, 'TPext'] * 0.043085)
            df.at[k, 'I_OPM_jma_tpcor'] = df.at[k, 'PO3_OPM'] * df.at[k, 'PFcor'] * JMA[-1] / \
                                    (df.at[k, 'Tpump_cor'] * 0.043085)
            df.at[k, 'JMA'] = JMA[-1]

        # ## komhyr corrections
        # for p in range(len(komhyr_en) - 1):
        #
        #     if df.at[k, 'ENSCI'] == 1:
        #         if (df.at[k, 'Pair'] >= Pval_komhyr[p + 1]) & (df.at[k, 'Pair'] < Pval_komhyr[p]):
        #             df.at[k, 'I_OPM_kom'] = df.at[k, 'PO3_OPM'] * df.at[k, 'PFcor'] * komhyr_en[p] / \
        #                                        (df.at[k, 'TPext'] * 0.043085)
        #     if df.at[k, 'ENSCI'] == 0:
        #         if (df.at[k, 'Pair'] >= Pval_komhyr[p + 1]) & (df.at[k, 'Pair'] < Pval_komhyr[p]):
        #             df.at[k, 'I_OPM_kom'] = df.at[k, 'PO3_OPM'] * df.at[k, 'PFcor'] * komhyr_sp[p] / \
        #                                        (df.at[k, 'TPext'] * 0.043085)
        #
        # if (df.at[k, 'Pair'] <= Pval_komhyr[-1]):
        #     if df.at[k, 'ENSCI'] == 1:
        #         df.at[k, 'I_OPM_kom'] = df.at[k, 'PO3_OPM'] * df.at[k, 'PFcor'] * komhyr_en[-1] / \
        #                                    (df.at[k, 'TPext'] * 0.043085)
        #     if df.at[k, 'ENSCI'] == 0:
        #         df.at[k, 'I_OPM_kom'] = df.at[k, 'PO3_OPM'] * df.at[k, 'PFcor'] * komhyr_sp[-1] / \
        #                                    (df.at[k, 'TPext'] * 0.043085)
        #

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

simlist = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Sim'])
teamlist = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Team'])
sol = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Sol'].tolist())
buff = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Buf'])
ensci = np.asarray(df.drop_duplicates(['Sim', 'Team'])['ENSCI'])

dft = {}
list_data = []

for j in range(len(simlist)):

    dft[j] = df[(df.Sim == simlist[j]) & (df.Team == teamlist[j])]
    dft[j] = dft[j].reset_index()

    for k in range(len(dft[j])):
        ## jma corrections

        dft[j]['PF'] = dft[j]['PFcor']
        # 2023 update:
        # ## komhyr corrections:
        if ensci[j] == 1:
            for i in range(len(komhyr_en) - 1):
                if (dft[j].at[k, 'Pair'] >= Pval_komhyr[i + 1]) & (dft[j].at[k, 'Pair'] < Pval_komhyr[i]):
                    # print(p, Pval[p + 1], Pval[p ])
                    dft[j].at[k, 'I_OPM_kom'] = dft[j].at[k, 'PO3_OPM'] * dft[j].at[k, 'PF'] * komhyr_en[i] / \
                                                (dft[j].at[k, 'TPext'] * 0.043085)
                    dft[j].at[k, 'I_OPM_kom_tpcor'] = dft[j].at[k, 'PO3_OPM'] * dft[j].at[k, 'PF'] * komhyr_en[i] / \
                                                (dft[j].at[k, 'Tpump_cor'] * 0.043085)

            if (dft[j].at[k, 'Pair'] <= Pval_komhyr[-1]):
                dft[j].at[k, 'I_OPM_kom'] = dft[j].at[k, 'PO3_OPM'] * dft[j].at[k, 'PF'] * komhyr_en[-1] / \
                                            (dft[j].at[k, 'TPext'] * 0.043085)
                dft[j].at[k, 'I_OPM_kom_tpcor'] = dft[j].at[k, 'PO3_OPM'] * dft[j].at[k, 'PF'] * komhyr_en[-1] / \
                                            (dft[j].at[k, 'Tpump_cor'] * 0.043085)

        if ensci[j] == 0:
            for i in range(len(komhyr_en) - 1):
                if (dft[j].at[k, 'Pair'] >= Pval_komhyr[i + 1]) & (dft[j].at[k, 'Pair'] < Pval_komhyr[i]):
                    # print(p, Pval[p + 1], Pval[p ])
                    dft[j].at[k, 'I_OPM_kom'] = dft[j].at[k, 'PO3_OPM'] * dft[j].at[k, 'PF'] * komhyr_sp[i] / \
                                                (dft[j].at[k, 'TPext'] * 0.043085)
                    dft[j].at[k, 'I_OPM_kom_tpcor'] = dft[j].at[k, 'PO3_OPM'] * dft[j].at[k, 'PF'] * komhyr_sp[i] / \
                                                (dft[j].at[k, 'Tpump_cor'] * 0.043085)

            if (dft[j].at[k, 'Pair'] <= Pval_komhyr[-1]):
                dft[j].at[k, 'I_OPM_kom'] = dft[j].at[k, 'PO3_OPM'] * dft[j].at[k, 'PF'] * komhyr_sp[-1] / \
                                            (dft[j].at[k, 'TPext'] * 0.043085)
                dft[j].at[k, 'I_OPM_kom_tpcor'] = dft[j].at[k, 'PO3_OPM'] * dft[j].at[k, 'PF'] * komhyr_sp[-1] / \
                                            (dft[j].at[k, 'Tpump_cor'] * 0.043085)

    list_data.append(dft[j])

df_dc2 = pd.concat(list_data, ignore_index=True)

# df_dc.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_deconv_updjma.csv")
df_dc2.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie1996_Data_2023.csv")