import numpy as np
import pandas as pd
# Libraries
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from data_cuts import cuts0910, cuts2017, cuts9602
from plotting_functions import filter_rdif, filter_rdif_all
from analyse_functions import Calc_average_Dif_yref, apply_calibration, cal_dif
from constant_variables import *
import warnings

def Calc_average_profile_pressure(dft, xcolumn):

    yref = [1000, 950, 900, 850, 800, 750, 700, 650, 600,  550, 500, 450,  400, 350, 325, 300, 275, 250, 240, 230, 220, 210, 200,
            190, 180, 170, 160, 150, 140, 130, 125, 120, 115, 110, 105,  100,95, 90, 85, 80, 75, 70, 65, 60, 55,
            50, 45, 40,  35, 30, 28, 26, 24, 22,  20, 19, 18, 17, 16, 15,  14, 13.5, 13, 12.5,  12, 11.5, 11, 10.5,
            10, 9.75, 9.50, 9.25, 9, 8.75, 8.5, 8.25,  8, 7.75, 7.5, 7.25,  7, 6.75, 6.50, 6.25, 6]

    n = len(yref) - 1
    Ygrid = [-9999.0] * n

    Xgrid = [-9999.0] * n
    Xsigma = [-9999.0] * n


    for i in range(n):
        dftmp1 = pd.DataFrame()
        dfgrid = pd.DataFrame()


        grid_min = yref[i+1]
        grid_max = yref[i]
        Ygrid[i] = (grid_min + grid_max) / 2.0

        filta = dft.Pair >= grid_min
        filtb = dft.Pair < grid_max
        filter1 = filta & filtb
        dftmp1['X'] = dft[filter1][xcolumn]

        filtnull = dftmp1.X > -9999.0
        dfgrid['X'] = dftmp1[filtnull].X

        Xgrid[i] = np.nanmean(dfgrid.X)
        Xsigma[i] = np.nanstd(dfgrid.X)

    return Xgrid, Xsigma, Ygrid



warnings.filterwarnings("ignore")

# plot style variables#
# size_label = 28
# size_title = 32
# size_tick = 26
# size_legend = 18
size_label = 36
size_title = 40
size_tick = 34
size_legend = 32
bool_sm_vh = True
pre = ''
if bool_sm_vh: pre = '_sm_hv'

bool_rdif = True
bool_two = True
bool_three = True
bool_triple = True
bool_adif = True
bool_test_2017 = False

df0910t = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_calibrated{pre}.csv", low_memory=False)

df0910 = df0910t.drop(['Unnamed: 0.1', 'index', 'Tact', 'Tair', 'Tinlet', 'TPint', 'I_Pump', 'VMRO3', 'VMRO3_OPM',
                       'ADif_PO3S', 'RDif_PO3S', 'Z', 'Header_Team', 'Header_Sim', 'Header_PFunc', 'Header_PFcor',
                       'Header_IB1', 'Simulator_RunNr', 'Date', 'Ini_Prep_Date', 'Prep_SOP', 'SerialNr', 'SerNr',
                       'Date_1',
                       'SondeAge', 'Solutions', 'Volume_Cathode', 'ByPass_Cell', 'Current_10min_after_noO3',
                       'Resp_Time_4_1p5_sec',
                       'RespTime_1minOver2min', 'Final_BG', 'T100', 'mlOvermin', 'T100_post', 'mlOverminp1',
                       'RespTime_4_1p5_sec_p1', 'RespTime_1minOver2min_microamps', 'PostTestSolution_Lost_gr',
                       'PumpMotorCurrent', 'PumpMotorCurrent_Post',
                       'PF_Unc', 'PF_Cor', 'BG', 'plog', 'Tcell', 'TcellC', 'Pw', 'massloss', 'Tboil', 'total_massloss',
                       'I_conv_slow',
                       'I_conv_slow_komhyr'], axis=1)

print(list(df0910))
df17 = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_calibrated{pre}.csv", low_memory=False)
df17 = df17[df17.iB2 >= 0]
df9602 = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie9602_calibrated{pre}.csv", low_memory=False)
df17['Year'] = '2017'
df0910['Year'] = '0910'
df9602['Year'] = '9602'

slist = [0, 1, 2, 3, 4, 5]
y = [0] * 6

bool_current = False

# year = '2017'
# year_title = '2017'
# pyear = "2017"

# year = '9602'
# year_title = '1996/1998/2000/2002'
# pyear = "9602"

year = '0910'
year_title = '2009/2010'
pyear = "0910"

# year = 'all'
#
df = df0910
slist = [0, 1, 3, 4]

bool_inter = True

if year == '2017':
    slist = [0, 2, 4, 5]
    df = df17
    df = df[df.iB2 >= 0]
    pre1 = f'scatter_'
    pre2 = f'scatter2_{pyear}_'
    if bool_test_2017:
        pre1 = 'test_'

if year == '9602':
    df = df9602
    pre1 = f'fig10_{pre}_'
    pre2 = f'fig13_{pre}_'

if year == 'all':
    df = pd.concat([df0910, df17], ignore_index=True)
    year_title = '2009/2010/2017'
    pyear = "0910-2017"
    pre = 'all_'
    slist = [0, 1, 2, 3, 4, 5]
    pre1 = f'scatter_{pre}_'
    pre2 = f'fig13plus_{pre}_'



df['PO3_cal'] = (0.043085 * df['Tpump_cor'] * df['I_corrected']) / (df['PFcor_jma'])
if year == '9602':
    df['PO3_cor'] = (0.043085 * df['Tpump_cor'] * df['Ifast_minib0_deconv_sm10']) / (df['PFcor_jma'])

if year == '0910' and bool_inter:
    snlist = ['PO3_dqa', 'PO3_OPM', 'PO3_cor', 'PO3_cal']
    clist = ['PO3_dqac', 'PO3_OPMc', 'PO3_corc', 'PO3_calc']
    vlist = ['Pair', 'ENSCI', 'Sol', 'Buf', 'Sim', 'Team']
    dfc = pd.DataFrame()
    dfc['Pair'] = df['Pair']
    dfc['ENSCI'] = df['ENSCI']
    dfc['Sol'] = df['Sol']
    dfc['Buf'] = df['Buf']
    dfc['Sim'] = df['Sim']
    dfc['Team'] = df['Team']

    for j in range(len(snlist)):
        dfc[clist[j]] = df[snlist[j]]

        dfc.loc[(dfc.Pair < 500) & (dfc.Pair > 350) & (dfc.ENSCI == 1), clist[j]] = np.nan
        dfc.loc[(dfc.Pair < 600) & (dfc.Pair > 300) & (dfc.ENSCI == 0), clist[j]] = np.nan

        dfc.loc[(dfc.Pair < 120) & (dfc.Pair > 57), clist[j]] = np.nan
        dfc.loc[(dfc.Pair < 29) & (dfc.Pair > 15), clist[j]] = np.nan

        dfc.loc[(dfc.Pair < 8) & (dfc.Pair > 4), clist[j]] = np.nan
    dfs = dfc[clist].interpolate()
    for v in vlist:
        dfs[v] = dfc[v]

    dfc = dfs.copy()
    for j in range(len(snlist)):
        dfc[snlist[j]] = 0
        dfc[snlist[j]] = dfs[clist[j]]


df = df[df.PO3_cor > 0]
df = df[df.PO3_cor < 99]

df = df[df.PO3_cal > 0]
df = df[df.PO3_cal < 99]


# df['ADif'], df['RDif'] = cal_dif(df, 'PO3_dqa', 'PO3_OPM', 'ADif', 'RDif')
# df['ADif_cor'], df['RDif_cor'] = cal_dif(df, 'PO3_cor', 'PO3_OPM', 'ADif_cor', 'RDif_cor')
# df['ADif_cal'], df['RDif_cal'] = cal_dif(df, 'PO3_cal', 'PO3_OPM', 'ADif_cal',
#                                          'RDif_cal')
profl = filter_rdif_all(df)
# print(profl[1])


if bool_inter:
    dfc = dfc[dfc.PO3_cor > 0]
    dfc = dfc[dfc.PO3_cor < 99]

    dfc = dfc[dfc.PO3_cal > 0]
    dfc = dfc[dfc.PO3_cal < 99]

    dfc = dfc[dfc.PO3_OPM > 0]
    dfc = dfc[dfc.PO3_OPM < 99]


    proflc = filter_rdif_all(dfc)


df0 = proflc[1]
print(list(df0))


opm, o3err, y = Calc_average_profile_pressure(df0, 'PO3_OPM')
o3, o3cerr, y = Calc_average_profile_pressure(df0, 'PO3_dqa')
o3trc, o3cerr, y = Calc_average_profile_pressure(df0, 'PO3_cor')
o3trcc, o3cerr, y = Calc_average_profile_pressure(df0, 'PO3_cal')

df1 = pd.DataFrame()
df1['y'] = y
df1['o3'] = o3
df1['opm'] = opm
df1['o3trc'] = o3trc
df1['o3trcc'] = o3trcc

int1 = int((3.9449 * (df1.opm.shift() + df1.opm) * np.log(df1.y.shift() / df1.y)).sum())
int2 = int((3.9449 * (df1.o3.shift() + df1.o3) * np.log(df1.y.shift() / df1.y)).sum())
int3 = int((3.9449 * (df1.o3trc.shift() + df1.o3trc) * np.log(df1.y.shift() / df1.y)).sum())
int4 = int((3.9449 * (df1.o3trcc.shift() + df1.o3trcc) * np.log(df1.y.shift() / df1.y)).sum())

print('o3 values', int1, int2, int3, int4)

fig, ax = plt.subplots(figsize=(17, 9))

ax.plot(opm, y,  label=f'OPM,TO={int1}', marker = 'o', markersize = 6)
ax.plot(o3, y,  label=f'Conventional,TO={int2}', marker = 's', markersize = 6)
ax.plot(o3trc, y,  label=f'TRC,TO={int3}', marker = 'd', markersize = 6)
ax.plot(o3trcc, y,  label=f'TRCC,TO={int4}', marker = 'p', markersize = 6)

# ax.plot(o3trc, y,  label = ' Raw ' + 'TO=' + str(int3), marker = 's', markersize = 6)

# ax.plot(o3c, y,  label = ' DQA ' + 'TO=' + str(int1), marker = 's', markersize = 6, linestyle='None')
# ax.plot(o3, y,  label = 'WOUDC ' + 'TO=' + str(int2), marker = 'o', markersize = 6, linestyle='None')

ax.set_ylim(1000, 5)
ax.set_yscale('log')
ax.legend(loc="best")
ax.set_ylabel('Pressure [hPa]')
ax.set_xlabel('PO3 [mPa]')
plt.title(labellist[1] + ' 2009-2010')

plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v8/en1010_0910.png')
# plt.savefig(path + 'Plots/' + Plotname + '.eps')
# plt.savefig(path + 'Plots/  ' + Plotname + '.pdf')

plt.show()


