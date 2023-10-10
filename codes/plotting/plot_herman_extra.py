# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator

from analyse_functions import Calc_average_Dif_yref, Calc_average_yref
from analyse_functions import calc_average_df_pressure, set_columns_nopair_dependence, cal_dif

from plotting_functions import errorPlot_ARDif_withtext, errorPlot_ARDif
from data_cuts import cuts0910, cuts2017, cuts9602
from plotting_functions import filter_rdif
from constant_variables import *
bool_noib0 = False
year = '2017'
year_0910=False
year_2017=True
year_9602=False

df = pd.read_csv(
        f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_deconv_2023paper_herman.csv", low_memory=False)

if year=='0910':
    df = cuts0910(df)
if year=='2017':
    df = cuts2017(df)

df['PO3_cor_original'] = (0.043085 * df['Tpump_cor'] * df['Ifast_minib0_deconv']) / (df['PFcor_jma'])
df['PO3_cor'] = (0.043085 * df['Tpump_cor'] * df['Ifasth_deconv']) / (df['PFcor_jma'])
df['PO3_islow'] = (0.043085 * df['Tpump_cor'] * df['Islow_conv']) / (df['PFcor_jma'])
df['PO3_ib0'] = (0.043085 * df['Tpump_cor'] * df['iB0']) / (df['PFcor_jma'])


df['PO3_OPM_div10'] = df['PO3_OPM']/10

if year_0910:
    slist=['PO3_dqa', 'PO3_OPM','PO3_cor_original','PO3_cor', 'Islow_conv','PO3_OPM_div10', 'PO3_islow','PO3_ib0']
    clist=['PO3_dqac', 'PO3_OPMc','PO3_cor_originalc', 'PO3_corc','Islow_convc','PO3_OPM_div10c', 'PO3_islowc','PO3_ib0c']
    vlist = ['Pair','ENSCI','Sol','Buf','Sim','Team']
    for j in range(len(slist)):
        df[clist[j]] = df[slist[j]]
        # df.loc[(df.Pair < 500) & (df.Pair > 350) & (df.ENSCI==1),clist[j]] = np.nan
        # df.loc[(df.Pair < 600) & (df.Pair > 300) & (df.ENSCI ==0),clist[j]] = np.nan
        #
        # df.loc[(df.Pair < 120) & (df.Pair > 57),clist[j]] = np.nan
        # df.loc[(df.Pair < 8) & (df.Pair > 4),clist[j]] = np.nan
        df.loc[(df.Pair < 500) & (df.Pair > 350) & (df.ENSCI == 1), clist[j]] = np.nan
        df.loc[(df.Pair < 600) & (df.Pair > 300) & (df.ENSCI == 0), clist[j]] = np.nan
        df.loc[(df.Pair < 120) & (df.Pair > 57), clist[j]] = np.nan
        df.loc[(df.Pair < 30) & (df.Pair > 7), clist[j]] = np.nan
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

adif_P_trrm, adif_P_trrm_err, rdif_P_trrm, rdif_P_trrm_err, Yp = \
    Calc_average_Dif_yref(prof, 'PO3_cor', 'PO3_OPM', 'pressure', yref)

adifo_P_trrm, adifo_P_trrm_err, rdifo_P_trrm, rdifo_P_trrm_err, Yp = \
    Calc_average_Dif_yref(prof, 'PO3_cor_original', 'PO3_OPM', 'pressure', yref)

islow_av,  Yp = \
    Calc_average_yref(prof, 'PO3_islow', 'PO3_OPM', 'pressure', yref)
ib0_av,  Yp = \
    Calc_average_yref(prof, 'PO3_ib0', 'PO3_OPM', 'pressure', yref)

opm_av,  Yp = \
    Calc_average_yref(prof, 'PO3_OPM_div10', 'PO3_OPM', 'pressure', yref)



ytitle = 'Pressure [hPa]'
# 2009
labellist = ['EN-SCI/SST0.5', 'EN-SCI/SST1.0', 'SPC/SST0.5', 'SPC/SST1.0']
labellistf = ['EN-SCI_SST0.5', 'EN-SCI_SST1.0', 'SPC_SST0.5', 'SPC_SST1.0']

# 2017
if year_2017:
    labellist = ['EN-SCI/SST0.5', 'EN-SCI/SST0.1', 'SPC/SST0.1', 'SPC/SST1.0']
    labellistf = ['EN-SCI_SST0.5', 'EN-SCI_SST0.1', 'SPC_SST0.1', 'SPC_SST1.0']

# ### Plotting

# First do the plotting using PO3
rxtitle = '(Sonde - OPM)/OPM [%] '
axtitle = 'Partial Ozone Pressure [mPa]'
# axtitlecur = r'[$\mu$A]'

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

# results = []
# for k in range(0, 10):
#         # adif_P_trrm, adif_P_trrm_err, rdif_P_trrm, rdif_P_trrm_err, Yp
#         result = Calc_average_Dif_yref(prof, f'PO3_cor_{k}', 'PO3_OPM', 'pressure', yref)
#         results.append(result)



size_label = 28
size_title = 32
size_tick = 26
size_legend = 18


for m in range(0,4):
        plt.close('all')

        fig, ax = plt.subplots(figsize=(11, 9))
        fig.subplots_adjust(bottom=0.17)
        plt.xlim([-2.5, 2.5])
        plt.ylim([1000, 5])
        plt.ylabel(ytitle, fontsize=size_label, labelpad=-15)
        plt.xlabel(axtitle, fontsize=size_label)

        plt.xticks(fontsize=size_tick)
        plt.yticks(fontsize=size_tick)
        plt.grid(True)
        plt.gca().tick_params(which='major', width=3)
        plt.gca().tick_params(which='minor', width=3)
        # plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
        plt.gca().xaxis.set_tick_params(length=5, which='minor')
        plt.gca().xaxis.set_tick_params(length=10, which='major')
        plt.gca().yaxis.set_tick_params(length=10, which='major')
        ax.axvline(x=0, color='grey', linestyle='--')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(useOffset=False)


        plt.plot(adif_P_trrm[m], Yp, label=f'Sonde - OPM (ifast deconv)')
        plt.plot(adifo_P_trrm[m], Yp, label=f'Sonde - OPM (trrm)')
        plt.plot(islow_av[m], Yp, label=r'P$_{Islow}$')
        plt.plot(ib0_av[m], Yp, label=r'P$_{iB0}$')

        plt.plot(opm_av[m], Yp, label=f'PO3 OPM/10')

        # plt.plot([],[], label=f'Ss original={betalist[m]}', linestyle='none')
        ax.legend(loc='best', frameon=True, fontsize=size_legend)


        plt.title(title + ' ' + labellist[m], fontsize=size_title)
        plotname = year + '_' + labellistf[m]
        print(plotname)
        plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v6/png/extraplot_v2_' + plotname + '.png')

        plt.show()
        plt.close()