import numpy as np
import pandas as pd
from scipy import stats
from data_cuts import cuts0910, cuts2017, cuts9602
from plotting_functions import filter_rdif_all
from analyse_functions import apply_calibration
from constant_variables import *
import warnings
warnings.filterwarnings("ignore")

#code to obtaion calibration functions of the all sonde solution types

def cal_dif(df, var1, var2, adif, rdif):

    df[adif] = df[var1] - df[var2]
    df[rdif] = (df[var1] - df[var2])/df[var2] * 100

    return df[adif], df[rdif]

def print_fit_variables(nametag, alist, alist_err, blist, blist_err):
    print(nametag)
    print('a=', alist)
    print('a_err=', alist_err)
    print('b=', blist)
    print('b_err=', blist_err)

write_to_df = True
bool_sm_vh = True

pre=''
if bool_sm_vh: pre = '_sm_hv'
df96c = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie9602_deconv_2023paper{pre}.csv", low_memory=False)
df96c = df96c[df96c.iB1 > -9]

df17c = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_deconv_2023paper_ib2{pre}.csv", low_memory=False)

df09c = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/"
                    f"Josie0910_deconv_2023_unitedpaper{pre}.csv",low_memory=False)


df09c['Ifast_minib0_deconv_sm10'] = df09c['Ifast_minib0_deconv_ib1_decay'].rolling(window=5, center=True).mean()
df96c['Ifast_minib0_deconv_sm10'] = df96c['Ifast_minib0_deconv'].rolling(window=5, center=True).mean()
df17c['Ifast_minib0_deconv_sm10'] = df17c['Ifast_minib0_deconv'].rolling(window=5, center=True).mean()
if bool_sm_vh:
    df09c['Ifast_minib0_deconv_sm10'] = df09c['Ifast_minib0_deconv_ib1_decay']
    df96c['Ifast_minib0_deconv_sm10'] = df96c['Ifast_minib0_deconv']
    df17c['Ifast_minib0_deconv_sm10'] = df17c['Ifast_minib0_deconv']

df09 = cuts0910(df09c)
df17 = cuts2017(df17c)
df96 = cuts9602(df96c)


df09['Year'] = '0910'
df17['Year'] = '2017'
df96['Year'] = '9602'

# dfa = pd.concat([df09, df17, df96], ignore_index=True)
dfa = pd.concat([df09, df17], ignore_index=True)

# year = '2017'
# year = '0910'

# year = '9602'
year = '0910/2017'
# df = dfa[dfa.Year == year]
# df = df96
df = dfa

df['ADif'], df['RDif'] = cal_dif(df, 'IM', 'I_OPM_kom', 'ADif', 'RDif')
df['ADif_cor'], df['RDif_cor'] = cal_dif(df, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma', 'ADif_cor', 'RDif_cor')

#now in terms of PO3
df['PO3_cor'] = (0.043085 * df['Tpump_cor'] * df['Ifast_minib0_deconv_sm10']) / (df['PFcor_jma'])
df['ADif_pre'], df['RDif_pre'] = cal_dif(df, 'PO3_dqa', 'PO3_OPM', 'ADif_pre', 'RDif_pre')
df['ADif_pre_cor'], df['RDif_pre_cor'] = cal_dif(df, 'PO3_cor', 'PO3_OPM', 'ADif_pre_cor', 'RDif_pre_cor')


prof = filter_rdif_all(df)

prof[1].to_excel("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_EN1010_file_crosscheck.xlsx")


print('stop')
slist = [0,1,2,3,4,5]
if year == '0910':slist = [0,1,3,4]
if year == '9602':slist = [0,1,3,4]
if year == '2017': slist = [0,2,4,5]

ls = len(slist)
ls = 6

nsim = [0] *ls
coeff = np.zeros((ls,6))
coeff_pre = np.zeros((ls,6))

varl = ['slope','intercept','rvalue','pvalue','stderr','intercept_stderr']

sign = ['+'] * len(prof)
labelc = [0] * len(prof)

rdif = [0] *ls
rdif_cor = [0] *ls
rdif_pre_cor = [0] *ls
y = [0] *ls
yp = [0] *ls

prdif = [0] *ls
prdif_cor = [0] *ls
prdif_pre_cor = [0] *ls

py = [0] *ls
pcut = 13
urdif = 30
lrdif = -30

alist = [0]*ls
alist_err = [0]*ls
blist = [0]*ls
blist_err = [0]*ls

alist_pre = [0]*ls
alist_pre_err = [0]*ls
blist_pre = [0]*ls
blist_pre_err = [0]*ls

for i in slist:

    print(i, slist)
    print(i, len(prof[i].drop_duplicates(['Sim', 'Team'])))
    nsim[i] = len(prof[i].drop_duplicates(['Sim', 'Team']))
    prof[i]['pair_nan'] = 0
    prof[i].loc[prof[i].Pair.isnull(), 'pair_nan'] = 1

    dft = prof[i][prof[i].pair_nan == 0]
    dfp = prof[i][prof[i].pair_nan == 0]

    prdif[i] = np.array(dft['RDif'])
    prdif_cor[i] = np.array(dft['RDif_cor'])
    prdif_pre_cor[i] = np.array(dfp['RDif_pre_cor'])

    py[i] = np.array(dft['Pair'])

    ####current #####
    dft = dft[(dft.RDif_cor < urdif) & (dft.RDif_cor > lrdif)]
    filt_p = dft.Pair >= pcut
    rdif[i] = np.array(dft[filt_p]['RDif'])
    rdif_cor[i] = np.array(dft[filt_p]['RDif_cor'])

    #### po3 #####
    dfp = dfp[(dfp.RDif_pre_cor < urdif) & (dfp.RDif_pre_cor > lrdif)]
    filt_p_pre = dfp.Pair >= pcut
    rdif_pre_cor[i] = np.array(dfp[filt_p_pre]['RDif_pre_cor'])

    y[i] = np.array(dft[filt_p]['Pair'])
    yp[i] = np.array(dfp[filt_p_pre]['Pair'])

    #fit using current
    res = stats.linregress(np.log10(y[i]), rdif_cor[i])
    #fit using po3_cor
    res_pre = stats.linregress(np.log10(yp[i]), rdif_pre_cor[i])

    for k in range(len(res)):
        coeff[i][k] = round(res[k],2)
        coeff_pre[i][k] = round(res_pre[k],2)
    coeff[i][5] = round(res.intercept_stderr,2)
    coeff_pre[i][5] = round(res_pre.intercept_stderr,2)

    alist[i] = coeff[i][1]
    alist_err[i] = coeff[i][5]
    blist[i] = coeff[i][0]
    blist_err[i] = coeff[i][4]

    alist_pre[i] = coeff_pre[i][1]
    alist_pre_err[i] = coeff_pre[i][5]
    blist_pre[i] = coeff_pre[i][0]
    blist_pre_err[i] = coeff_pre[i][4]

    #[slope, intercept, rvalue, pvalue, stderr, intercept_err]
    labelc[i] = coeff[i][0]

    if coeff[i][0] < 0:
        sign[i] = '-'
        labelc[i] = -1 * coeff[i][0]
        labelc[i] = -1 * coeff[i][0]


print_fit_variables(f'w.r.t. current - {year}', alist, alist_err, blist, blist_err)
print_fit_variables(f'w.r.t. po3 - {year}', alist_pre, alist_pre_err, blist_pre, blist_pre_err)



#now apply calibration functions
###########################################################3
#################################################################
######################################################
#now apply calibration functions


# df = df96

if write_to_df:

    slist = [0, 1, 3, 4]
    year = '2017'
    if year == '2017': slist = [0, 2, 4, 5]
    df = dfa[dfa.Year == year]
    df['IminusiB1'] = df['IM'] - df['iB2']


    prof_cor = apply_calibration(df, coeff)

    df_cor = pd.concat([prof_cor[0], prof_cor[2], prof_cor[4], prof_cor[5]], ignore_index=True)
    df_cor['ADif'], df_cor['RDif'] = cal_dif(df_cor, 'IminusiB1', 'I_OPM_kom', 'ADif', 'RDif')
    df_cor['ADif_cor'], df_cor['RDif_cor'] = cal_dif(df_cor, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma', 'ADif_cor', 'RDif_cor')
    df_cor['ADif_cal'], df_cor['RDif_cal'] = cal_dif(df_cor, 'I_corrected', 'I_OPM_jma', 'ADif_cal',
                                                               'RDif_cal')
    df_cor.to_csv(f'/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_calibrated{pre}.csv')

    ######
    year = '0910'
    slist = [0,1,3,4]
    if year == '2017': slist = [0,2,4,5]
    df = dfa[dfa.Year == year ]
    # df = df96
    df['IminusiB1'] = df['IM'] - df['iB1']

    prof_cor = apply_calibration(df, coeff)

    df_cor = pd.concat([prof_cor[0], prof_cor[1], prof_cor[3], prof_cor[4]], ignore_index=True)
    df_cor['ADif'], df_cor['RDif'] = cal_dif(df_cor, 'IminusiB1', 'I_OPM_kom', 'ADif', 'RDif')
    df_cor['ADif_cor'], df_cor['RDif_cor'] = cal_dif(df_cor, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma', 'ADif_cor', 'RDif_cor')
    df_cor['ADif_cal'], df_cor['RDif_cal'] = cal_dif(df_cor, 'I_corrected', 'I_OPM_jma', 'ADif_cal',
                                                               'RDif_cal')
    df_cor.to_csv(f'/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_calibrated{pre}.csv')

    ######
    year = '9602'
    slist = [0, 1, 3, 4]
    if year == '2017': slist = [0, 2, 4, 5]
    df = df96
    df['IminusiB1'] = df['IM'] - df['iB1']

    prof_cor = apply_calibration(df, coeff)

    df_cor = pd.concat([prof_cor[0], prof_cor[1], prof_cor[3], prof_cor[4]], ignore_index=True)
    df_cor['ADif'], df_cor['RDif'] = cal_dif(df_cor, 'IminusiB1', 'I_OPM_kom', 'ADif', 'RDif')
    df_cor['ADif_cor'], df_cor['RDif_cor'] = cal_dif(df_cor, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma', 'ADif_cor',
                                                     'RDif_cor')
    df_cor['ADif_cal'], df_cor['RDif_cal'] = cal_dif(df_cor, 'I_corrected', 'I_OPM_jma', 'ADif_cal',
                                                     'RDif_cal')
    df_cor.to_csv(f'/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_calibrated{pre}.csv')






