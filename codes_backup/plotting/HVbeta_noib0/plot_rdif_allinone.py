# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator

from analyse_functions import Calc_average_Dif_yref, Calc_average_Dif_yref_df
from plotting_functions import errorPlot_ARDif_withtext, errorPlot_ARDif
from data_cuts import cuts0910, cuts2017, cuts9602
from plotting_functions import filter_rdif
from constant_variables import *

folderpath = ''

year_2017 = True
year_9602 = False
year_0910 = False
bool_inter = False

# tag = '10percent_interp'
tag = '20percent_HVMethod_'

bool_sm_vh = True
if bool_sm_vh:
    pre = '_sm_hv'
    end = ''
else:
    pre = ''

bool_current = False
bool_pressure = True
if year_9602:
    df = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie9602_calibrated{pre}.csv", low_memory=False)

if year_2017:
    # df = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_calibrated{pre}.csv", low_memory=False)
    df = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_calibrated_HVbeta_sm_hv.csv", low_memory=False)


if year_0910:
    df = pd.read_csv(
        f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_calibrated_HVbeta_sm_hv.csv", low_memory=False)
    print(list(df))

if year_9602 or year_2017:
    df['Ifast_minib0_deconv_sm10'] = df['Ifast_deconv'].rolling(window=5, center=True).mean()
    if bool_sm_vh:
        df['Ifast_minib0_deconv_sm10'] = df['Ifast_deconv']
        # df['Ifast_minib0_deconv_sm10'] = df['Ifast_deconv']
        df['PO3_cor'] = (0.043085 * df['Tpump_cor'] * df['Ifast_minib0_deconv_sm10']) / (df['PFcor_jma'])

if year_0910:
    df['Ifast_minib0_deconv_sm10'] = df['Ifast_deconv_ib1_decay'].rolling(window=5, center=True).mean()
    if bool_sm_vh:
        df['Ifast_minib0_deconv_sm10'] = df['Ifast_deconv_ib1_decay']
        # df['Ifast_minib0_deconv_sm10'] = df['Ifast_deconv_ib1_decay']
        df['PO3_cor'] = (0.043085 * df['Tpump_cor'] * df['Ifast_minib0_deconv_sm10']) / (df['PFcor_jma'])

# if year_0910:
#     df = cuts0910(df)
# if year_2017:
#     df = cuts2017(df)
# if year_9602:
#     df = cuts9602(df)

df['IminusiB1'] = df['IM'] - df['iB1']
if year_2017:
    df['IminusiB1'] = df['IM'] - df['iB2']
    filt_01 = (df.Buf == 0.1)
    not_filt_01 = (df.Buf != 0.1)
    df.loc[filt_01, 'I_OPM_kom'] = (df.loc[filt_01, 'PO3_OPM'] * df.loc[filt_01, 'PFcor_jma']) / \
                                   (df.loc[filt_01, 'Tpump_cor'] * 0.043085)
    df.loc[not_filt_01, 'I_OPM_kom'] = (df.loc[not_filt_01, 'PO3_OPM'] * df.loc[not_filt_01, 'PFcor_kom']) / \
                                       (df.loc[not_filt_01, 'Tpump_cor'] * 0.043085)

    # df.loc[filt_01, 'PO3_dqa'] = (0.043085 * df.loc[filt_01, 'Tpump_cor'] * (
    #             df.loc[filt_01, 'IM'] - df.loc[filt_01, 'iB2'])) \
    #                              / (df.loc[filt_01, 'PFcor_jma'])
    # df.loc[not_filt_01, 'PO3_dqa'] = (0.043085 * df.loc[not_filt_01, 'Tpump_cor'] * (
    #             df.loc[not_filt_01, 'IM'] - df.loc[not_filt_01, 'iB2'])) \
    #                                  / (df.loc[not_filt_01, 'PFcor_kom'])

df['PO3_cal'] = (0.043085 * df['Tpump_cor'] * df['I_corrected']) / (df['PFcor_jma'])
if year_9602:
    df['PO3_cor'] = (0.043085 * df['Tpump_cor'] * df['Ifast_minib0_deconv_sm10']) / (df['PFcor_jma'])


if year_0910 and bool_inter:
    slist=['PO3_dqa', 'PO3_OPM', 'PO3_cor','PO3_cal']
    clist=['PO3_dqac', 'PO3_OPMc', 'PO3_corc','PO3_calc']
    vlist = ['Pair','ENSCI','Sol','Buf','Sim','Team']
    for j in range(len(slist)):
        df[clist[j]] = df[slist[j]]
        # df.loc[(df.Pair < 475) & (df.Pair > 375),clist[j]] = np.nan
        # df.loc[(df.Pair < 100) & (df.Pair > 85),clist[j]] = np.nan
        # df.loc[(df.Pair < 6) & (df.Pair > 5),clist[j]] = np.nan
        #interp1
        # df.loc[(df.Pair < 500) & (df.Pair > 350),clist[j]] = np.nan
        # df.loc[(df.Pair < 120) & (df.Pair > 57),clist[j]] = np.nan
        # df.loc[(df.Pair < 8) & (df.Pair > 4),clist[j]] = np.nan
        #interp2
        df.loc[(df.Pair < 500) & (df.Pair > 350) & (df.ENSCI==1),clist[j]] = np.nan
        df.loc[(df.Pair < 600) & (df.Pair > 300) & (df.ENSCI ==0),clist[j]] = np.nan

        df.loc[(df.Pair < 120) & (df.Pair > 57),clist[j]] = np.nan
        df.loc[(df.Pair < 8) & (df.Pair > 4),clist[j]] = np.nan
    dfs = df[clist].interpolate()
    for v in vlist:
        dfs[v] = df[v]

    df = dfs.copy()
    for j in range(len(slist)):
        df[slist[j]] = 0
        df[slist[j]] = dfs[clist[j]]


#     # (RT1: 475-375 hPa, RT2: 100-85hPa, RT3: 20-15 hPa, RT4: 6-5 hPa)


###############################################
# Filters for Sonde, Solution, Buff er selection
prof = filter_rdif(df, year_9602, year_0910, year_2017)

print(list(df))
# ##################################################################################
# ################     PLOTS        #################################
# ##################################################################################

ytitle = 'Pressure [hPa]'
# 2009
labellist = ['EN-SCI SST0.5', 'EN-SCI SST1.0', 'SPC SST0.5', 'SPC SST1.0']
# 2017
if year_2017:
    labellist = ['EN-SCI SST0.5', 'EN-SCI SST0.1', 'SPC SST0.1', 'SPC SST1.0']

# ### Plotting

# First do the plotting using PO3
rxtitle = '(Sonde - OPM)/OPM [%]'
# axtitle = 'Sonde - OPM [mPa]'
axtitlecur = r'Sonde - OPM [$\mu$A]'

title = 'JOSIE 0910'
if year_2017:
    title = 'JOSIE 2017'
    y = '2017'
if year_9602:
    y = '9602'
    title = 'JOSIE 1996/1998/2000/2002'
if year_0910:
    y = '0910'
    title = 'JOSIE 2009/2010'


ptitle = f' {title} Conventional'
ptitle_trrm = f' {title} \n TRC & No'+r' I$_{B0}$ & S$_S$ by Vömel et al.(2020)  '
ptitle_cal = f' {title} TRC + Calibrated'



# ##################################################################################
# # ################      Pressure PLOTS        #################################
# # ##################################################################################
# standard for all years
adif_P, adif_P_err, rdif_P, rdif_P_err, Yp = Calc_average_Dif_yref_df(df, 'PO3_dqa', 'PO3_OPM', 'pressure', yref)
adif_P_trrm, adif_P_trrm_err, rdif_P_trrm, rdif_P_trrm_err, Yp = \
    Calc_average_Dif_yref_df(df, 'PO3_cor', 'PO3_OPM', 'pressure', yref)
adif_P_cal, adif_P_cal_err, rdif_P_cal, rdif_P_cal_err, Yp = \
    Calc_average_Dif_yref_df(df, 'PO3_cal', 'PO3_OPM', 'pressure', yref)




nsim = len(df.drop_duplicates(['Sim', 'Team']))
print('nsim', nsim)

# tag = 'interp2_'
adifc = f'{tag}ADif_{y}_pressure_conventional{pre}{end}'
adift_c = f'{tag}ADif_{y}_pressure_trrm{pre}{end}'
adifc_c = f'{tag}ADif_{y}_pressure_calibrated{pre}{end}'
rdifc = f'{tag}RDif_{y}_pressure_conventional{pre}{end}'
rdift_c = f'{tag}RDif_{y}_pressure_trrm{pre}{end}'
rdifc_c = f'{tag}RDif_{y}_pressure_calibrated{pre}{end}'

axtitlepre = r'Sonde - OPM [mPa]'

colorl = ['#e41a1c',  '#dede00',  '#984ea3', '#377eb8']

# errorPlot_ARDif_withtext(rdif_P, rdif_P_err, Yp, [-40, 40], [1000, 5], ptitle,
#                          rxtitle, ytitle, labellist, rdifc, nsim, True, 1)
#
# errorPlot_ARDif_withtext(rdif_P_trrm, rdif_P_trrm_err, Yp, [-40, 40], [1000, 5], ptitle_trrm,
#                          rxtitle, ytitle, labellist, rdift_c, nsim, True, 1)
#
# errorPlot_ARDif_withtext(rdif_P_cal, rdif_P_cal_err, Yp, [-40, 40], [1000, 5], ptitle_cal,
#                          rxtitle, ytitle, labellist, rdifc_c, nsim, True, 1)

## y locations of the extra text which is drawn for the total O3 values
texty = [0.23, 0.16, 0.09, 0.02]
size_label = 28
size_title = 32
size_tick = 26
size_legend = 18

plt.close('all')
fig, ax = plt.subplots(figsize=(11, 9))
# plt.figure(figsize=(12, 8))


fig.subplots_adjust(bottom=0.17)
# ax.yaxis.set_major_formatter(ScalarFormatter())

plt.xlim( [-15, 10])
plt.ylim( [1000, 5])
# plt.title(maintitle, fontsize=size_title)
plt.xlabel(rxtitle, fontsize=size_label)
plt.ylabel(ytitle, fontsize=size_label, labelpad=-15)
plt.xticks(fontsize=size_tick)
plt.yticks(fontsize=size_tick)
plt.grid(True)
# plt.gca().yaxis.set_major_formatter(ScalarFormatter())

plt.gca().tick_params(which='major', width=3)
plt.gca().tick_params(which='minor', width=3)
# plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().xaxis.set_tick_params(length=5, which='minor')
plt.gca().xaxis.set_tick_params(length=10, which='major')
plt.gca().yaxis.set_tick_params(length=10, which='major')

# ax.set_xticklabels(['',-20,'','', -10,'', 0,'', 10,'', 20])

# plt.yticks(np.arange(0, 7001, 1000))

# reference line
ax.axvline(x=0, color='grey', linestyle='--')
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(useOffset=False)

ax.plot([], [], ' ', label=r"N$_{sim}$" + f"={nsim}")#'

ax.errorbar(rdif_P[0], Yp, xerr=rdif_P_err[0], label='Conventional', color='#377eb8', linewidth=2,
                elinewidth=0.5, capsize=1, capthick=0.5)
ax.errorbar(rdif_P_trrm[0], Yp, xerr=rdif_P_trrm_err[0], label=r'TRC & No'+r' I$_{B0}$ & S$_S$ by Vömel et al.(2020)',
            color='#e41a1c', linewidth=2,
                elinewidth=0.5, capsize=1, capthick=0.5)

    # if textbool: tl[i] = ax.text(0.05, texty[i], textl[i], color=colorl[i], transform=ax.transAxes)


ax.legend(loc='best', frameon=True, fontsize=size_legend)

plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v5/png/Rdif_allinone_nocut' + '.png')
# plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v5/pdf/Rdif_allinone'+  '.pdf')
# plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v5/eps/Rdif_allinone'+  '.eps')
plt.show()


plt.close()









####################
# ##################################################################################
# # ################      CURRENT IM PLOTS        #################################
# # ##################################################################################
# standard for all years
