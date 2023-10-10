import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import glob

allFiles = glob.glob("/home/poyraden/Analysis/JOSIEfiles/Josie_0910_Preparation/*.asc")

list_data = []

#Some declarations

columnString = "Tact Tsys Temp_TMC_Offset Temp_TMC_Ist_7 I_1 I_2 I_3 I_4"
columnStr = columnString.split(" ")

for filename in allFiles:
    simnumber = int(filename.split("/SI")[1][0:3])
    df = pd.read_csv(filename, sep="\t", engine="python", skiprows=7, names=columnStr)
    df['Sim'] = simnumber
    list_data.append(df)

df = pd.concat(list_data, ignore_index=True)
df.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_preparation.csv")


# dfs = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Data_updjma.csv", low_memory=False)
dfs = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Data_2023paper.csv", low_memory=False)



dfs['timeall'] = pd.to_datetime(dfs['Tact'], format='%H:%M:%S').dt.time
# .dt.time
dfs['hour'] = pd.to_datetime(dfs['Tact'], format='%H:%M:%S').dt.hour
dfs['minute'] = pd.to_datetime(dfs['Tact'], format='%H:%M:%S').dt.minute
dfs['second'] = pd.to_datetime(dfs['Tact'], format='%H:%M:%S').dt.second
dfs['pctime_seconds'] = dfs['hour'] * 60 * 60 + dfs['minute']*60 + dfs['second']
dfs['Tact'] = dfs['pctime_seconds']

dfs = dfs.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Tair', 'I_Pump', 'VMRO3', 'VMRO3_OPM', 'ADif_PO3S', 'RDif_PO3S', 'Z', 'Header_Team',
        'Header_Sim', 'Header_PFunc', 'Header_PFcor', 'Header_IB1',  'Code', 'Flow', 'IB1', 'Cor', 'Simib', 'Teamib', 'Yearib', 'Simm', 'Teamm', 'Mspre', 'Mspost', 'Diff',
     'I_conv_slow', 'I_conv_slow_komhyr',  'hour', 'minute', 'second', 'timeall','Tinlet', 'TPint', 'pctime_seconds'], axis = 1)



simlist2 = df.drop_duplicates(['Sim'])['Sim'].tolist()
print('simlist2', simlist2)

s = len(simlist2)
simlist = simlist2 + simlist2 + simlist2 + simlist2

t1 = [1] * s
t2 = [2] * s
t3 = [3] * s
t4 = [4] * s

teamlist = t1 + t2 + t3 + t4
print('len(simlist), len(teamlist)',len(simlist), len(teamlist))


listdata_all = []

for j in range(len(simlist)):

    df3 = dfs[(dfs.Sim == simlist[j]) & (dfs.Team == teamlist[j])] # simulation data
    df1 = df[df.Sim == simlist[j]] #preparation data

    df3['TimeBool'] = 1
    df1['TimeBool'] = 0

    df3['TimeTag'] = 'Sim'
    df1['TimeTag'] = 'Prep'

    #     print(simlist[j], len(df1))
    if len(df1) == 0: continue
    if len(df3) == 0: continue

    if (teamlist[j] == 1):
        # print(teamlist[j])
        df1 = df1.drop(['Temp_TMC_Offset', 'Temp_TMC_Ist_7', 'I_2', 'I_3', 'I_4'], axis=1)
        df1['IM'] = df1['I_1']
        df1['Team'] = 1
        df1 = df1.drop(['I_1'], axis=1)
    if (teamlist[j] == 2):
        df1 = df1.drop(['Temp_TMC_Offset', 'Temp_TMC_Ist_7', 'I_1', 'I_3', 'I_4'], axis=1)
        df1['IM'] = df1['I_2']
        df1 = df1.drop(['I_2'], axis=1)
        df1['Team'] = 2
    if (teamlist[j] == 3):
        df1 = df1.drop(['Temp_TMC_Offset', 'Temp_TMC_Ist_7', 'I_1', 'I_2', 'I_4'], axis=1)
        df1['IM'] = df1['I_3']
        df1 = df1.drop(['I_3'], axis=1)
        df1['Team'] = 3
    if (teamlist[j] == 4):
        df1 = df1.drop(['Temp_TMC_Offset', 'Temp_TMC_Ist_7', 'I_1', 'I_3', 'I_2'], axis=1)
        df1['IM'] = df1['I_4']
        df1 = df1.drop(['I_4'], axis=1)
        df1['Team'] = 4


    eind1 = df1.last_valid_index()
    bind3 = df3.first_valid_index()
    # print(simlist[j], eind1, type(eind1), teamlist[j])
    df1['Tact'] = df1['Tact'].astype(int)
    df3['Tact'] = df3['Tact'].astype(int)

    end1 = df1.at[df1.last_valid_index(), 'Tact']
    begin3 = df3.at[df3.first_valid_index(), 'Tact']

    df2 = pd.DataFrame()

    arrtact = [i for i in range(end1 + 2, begin3, 2)]
    df2['Tact'] = arrtact
    df2['IM'] = 0
    df2['I_OPM_jma'] = 0
    df2['I_OPM_kom'] = 0
    df2['Sim'] = simlist[j]
    df2['Team'] = teamlist[j]
    df2['iB0'] = df3.at[bind3,'iB0']
    df2['iB1'] = df3.at[bind3,'iB1']
    df2['iB2'] = df3.at[bind3,'iB2']
    df2['ENSCI'] = df3.at[bind3,'ENSCI']
    df2['Sol'] = df3.at[bind3,'Sol']
    df2['Buf'] = df3.at[bind3,'Buf']
    df2['ADX'] = df3.at[bind3,'ADX']
    df2['TimeBool'] = 0
    df2['TimeTag'] = 'Between'


    df1['I_OPM_kom'] = 0
    df1['I_OPM_jma'] = 0
    # df1['Sim'] = simlist[j]
    # df1['Team'] = teamlist[j]
    df1['iB0'] = df3.at[bind3, 'iB0']
    df1['iB1'] = df3.at[bind3, 'iB1']
    df1['iB2'] = df3.at[bind3, 'iB2']
    df1['ENSCI'] = df3.at[bind3, 'ENSCI']
    df1['Sol'] = df3.at[bind3, 'Sol']
    df1['Buf'] = df3.at[bind3, 'Buf']
    df1['ADX'] = df3.at[bind3, 'ADX']


    frame = [df1, df2, df3]
    dffinal = pd.concat(frame)

    if ((simlist[j] == 138) | (simlist[j] == 140) | ((simlist[j] == 158))):
        start = dffinal.at[dffinal.first_valid_index(), 'Tact']
        maxt = dffinal.Tact.max()
        dif = maxt - start
        print(simlist[j], start, maxt, dif)

        dffinal['Tact'] = dffinal['Tact'].apply(lambda x: x - start if (x >= start) else x + dif)

    dffinal = dffinal.reset_index()


    zero = pd.Timestamp(dffinal.at[0, 'Tact'], unit='s')

    for i in range(len(dffinal)):
        dffinal.at[i, 'Time_ts'] = pd.Timestamp(dffinal.at[i, 'Tact'], unit='s')
        #     one = pd.Timestamp(dft.at[i, 'Tsim'], unit ='s')
        two = pd.Timestamp(dffinal.at[i, 'Tact'], unit='s')
        dffinal.at[i, 'Time_all'] = (two - zero).total_seconds()

    listdata_all.append(dffinal)

dfall = pd.concat(listdata_all, ignore_index=True)

dfall['Tsim_original'] = dfall['Tsim']
dfall['Tsim_pctime'] = dfall['Tact']
dfall['Tsim'] = dfall['Time_all']

# dfall = dfall.drop(['Temp_TMC_Offset', 'Temp_TMC_Ist_7', 'I_1', 'I_2', 'I_3', 'I_4'], axis = 1)


# dfall.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_preparationadded_updjma.csv")
dfall.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_preparationadded_2023paper.csv")

