import numpy as np
import pandas as pd
from scipy import stats
# Libraries
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from data_cuts import cuts0910, cuts2017, cuts9602
from plotting_functions import filter_rdif, filter_rdif_all
from analyse_functions import Calc_average_Dif_yref,apply_calibration
import warnings
warnings.filterwarnings("ignore")

#code to make final fitting plots of the all sonde solution types

def cal_dif(df, var1, var2, adif, rdif):

    df[adif] = df[var1] - df[var2]
    df[rdif] = (df[var1] - df[var2])/df[var2] * 100

    return df[adif], df[rdif]

def func1(y, a, b):
    return a + b*np.log10(y)

df96c = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie9602_deconv_2023.csv", low_memory=False)

df96c = df96c[df96c.iB1 > -9]

# df17c = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_deconv_tfast_upd.csv", low_memory=False)
df17c = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_deconv_2023paper.csv", low_memory=False)


df09c = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/"
                    "Josie0910_deconv_2023_unitedpaper.csv",low_memory=False)

df09c['Ifast_minib0_deconv_sm10'] = df09c['Ifast_minib0_deconv_ib1_decay'].rolling(window=5, center=True).mean()

df96c['Ifast_minib0_deconv_sm10'] = df96c['Ifast_minib0_deconv'].rolling(window=5, center=True).mean()
df17c['Ifast_minib0_deconv_sm10'] = df17c['Ifast_minib0_deconv'].rolling(window=5, center=True).mean()

df09 = cuts0910(df09c)
df17 = cuts2017(df17c)
df96 = cuts9602(df96c)

df09['Year'] = '0910'
df17['Year'] = '2017'
df96['Year'] = '9602'

# dfa = pd.concat([df09, df17, df96], ignore_index=True)
dfa = pd.concat([df09, df17], ignore_index=True)

year = '2009,2010,2017'
# year = '0910'
# year = '1998'
# df = dfa[dfa.Year == year]
df = dfa

df['ADif'], df['RDif'] = cal_dif(df, 'IM', 'I_OPM_kom', 'ADif', 'RDif')
df['ADif_cor'], df['RDif_cor'] = cal_dif(df, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma', 'ADif_cor', 'RDif_cor')

yref = [1000, 850, 700, 550, 400, 350, 325, 300, 250, 200, 175, 150, 125, 100, 87, 75, 67, 50, 42, 35, 30, 25, 22.5,
        20, 17.5, 15, 10, 8, 6]


print('len df', len(df))

prof = filter_rdif_all(df)
print(len(prof))

labellist = ['EN-SCI SST0.5', 'EN-SCI SST1.0', 'EN-SCI SST0.1','SPC SST0.5', 'SPC SST1.0', 'SPC SST0.1']
labellist_title = ['EN-SCI 0.5%', 'EN-SCI 1.0%', 'EN-SCI 0.1%','SPC 0.5%', 'SPC 1.0%', 'SPC 0.1%']
cbl = ['#e41a1c', '#a65628','#dede00', '#4daf4a', '#377eb8', '#984ea3']


slist = [0,1,2,3,4,5]
nsim = [0] * 6

coeff = np.zeros((6,6))
varl = ['slope','intercept','rvalue','pvalue','stderr','intercept_stderr']

sign = ['+'] * len(prof)
labelc = [0] * len(prof)

rdif = [0] * 6
rdif_cor = [0] * 6
y = [0] * 6

prdif = [0] * 6
prdif_cor = [0] * 6
py = [0] * 6
pcut = 13
urdif = 30
lrdif = -30


for i in slist:

    print(i, slist)
    nsim[i] = len(prof[i].drop_duplicates(['Sim', 'Team']))
    prof[i]['pair_nan'] = 0
    prof[i].loc[prof[i].Pair.isnull(), 'pair_nan'] = 1

    dft = prof[i][prof[i].pair_nan == 0]

    # dft['rdif_nan'] = 0
    # dft.loc[dft.RDif_cor.isnull(), 'rdif_nan'] = 1
    # dft = dft[dft.rdif_nan == 0]
    # dft = dft.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    prdif[i] = np.array(dft['RDif'])
    prdif_cor[i] = np.array(dft['RDif_cor'])
    py[i] = np.array(dft['Pair'])

    ####test dft#####
    dft = dft[(dft.RDif_cor < urdif) & (dft.RDif_cor > lrdif)]
    filt_p = dft.Pair >= pcut

    rdif[i] = np.array(dft[filt_p]['RDif'])
    rdif_cor[i] = np.array(dft[filt_p]['RDif_cor'])

    y[i] = np.array(dft[filt_p]['Pair'])
    print('rdif_cor[i]', rdif_cor[i])
    res = stats.linregress(np.log10(y[i]), rdif_cor[i])
    print('lin reg', res)
    print('in reg two', res.slope, res[0])
    for k in range(len(res)):
        coeff[i][k] = round(res[k],2)
        print('coeff[i][k]',i, k, coeff[i][k])
    coeff[i][5] = round(res.intercept_stderr,2)
    print('coeff[i][5]',i, '5', coeff[i][5])

    #[slope, intercept, rvalue, pvalue, stderr, intercept_err]
    labelc[i] = coeff[i][0]

    if coeff[i][0] < 0:
        sign[i] = '-'
        labelc[i] = -1 * coeff[i][0]
        labelc[i] = -1 * coeff[i][0]


cbl = ['#e41a1c', '#a65628','#dede00', '#4daf4a', '#377eb8', '#984ea3']

alist = [0]*6
alist_err = [0]*6
blist = [0]*6
blist_err = [0]*6

for k in range(len(labellist)):
    print(labellist[k], f'a = {coeff[k][1]} + b={coeff[k][0]}')

    alist[k] = coeff[k][1]
    alist_err[k] = coeff[k][5]
    blist[k] = coeff[k][0]
    blist_err[k] = coeff[k][4]


print('a=', alist)
print('a_err=', alist_err)
print('b=', blist)
print('b_err=', blist_err)
#now apply calibration functions
###########################################################3
#################################################################
######################################################
#now apply calibration functions

slist = [0,1,3,4]
year = '2017'
if year == '2017': slist = [0,2,4,5]
df = dfa[dfa.Year == year ]
# df = df96
# df['ADif'], df['RDif'] = cal_dif(df, 'IM', 'I_OPM_kom', 'ADif', 'RDif')
# df['ADif_cor'], df['RDif_cor'] = cal_dif(df, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma', 'ADif_cor', 'RDif_cor')
# profl = filter_rdif_all(df)
#
# adif_IM, adif_IM_err, rdif_IM, rdif_IM_err, Yp = Calc_average_Dif_yref(profl, 'IM', 'I_OPM_kom','pressure', yref)
# adif_IM_deconv10, adif_IM_deconv10_err, rdif_IM_deconv10, rdif_IM_deconv10_err, Yp = \
#     Calc_average_Dif_yref(profl, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma','pressure', yref)

prof_cor = apply_calibration(df, coeff)

df_cor = pd.concat([prof_cor[0], prof_cor[1], prof_cor[3], prof_cor[4]], ignore_index=True)
df_cor['ADif'], df_cor['RDif'] = cal_dif(df_cor, 'IM', 'I_OPM_kom', 'ADif', 'RDif')
df_cor['ADif_cor'], df_cor['RDif_cor'] = cal_dif(df_cor, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma', 'ADif_cor', 'RDif_cor')
df_cor['ADif_cal'], df_cor['RDif_cal'] = cal_dif(df_cor, 'I_corrected', 'I_OPM_jma', 'ADif_cal',
                                                           'RDif_cal')
df_cor.to_csv(f'/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_calibrated.csv')

######
year = '0910'
slist = [0,1,3,4]
if year == '2017': slist = [0,2,4,5]
df = dfa[dfa.Year == year ]
# df = df96
# df['ADif'], df['RDif'] = cal_dif(df, 'IM', 'I_OPM_kom', 'ADif', 'RDif')
# df['ADif_cor'], df['RDif_cor'] = cal_dif(df, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma', 'ADif_cor', 'RDif_cor')
# profl = filter_rdif_all(df)
#
# adif_IM, adif_IM_err, rdif_IM, rdif_IM_err, Yp = Calc_average_Dif_yref(profl, 'IM', 'I_OPM_kom','pressure', yref)
# adif_IM_deconv10, adif_IM_deconv10_err, rdif_IM_deconv10, rdif_IM_deconv10_err, Yp = \
#     Calc_average_Dif_yref(profl, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma','pressure', yref)

prof_cor = apply_calibration(df, coeff)

df_cor = pd.concat([prof_cor[0], prof_cor[1], prof_cor[3], prof_cor[4]], ignore_index=True)
df_cor['ADif'], df_cor['RDif'] = cal_dif(df_cor, 'IM', 'I_OPM_kom', 'ADif', 'RDif')
df_cor['ADif_cor'], df_cor['RDif_cor'] = cal_dif(df_cor, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma', 'ADif_cor', 'RDif_cor')
df_cor['ADif_cal'], df_cor['RDif_cal'] = cal_dif(df_cor, 'I_corrected', 'I_OPM_jma', 'ADif_cal',
                                                           'RDif_cal')
df_cor.to_csv(f'/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_calibrated.csv')




