import numpy as np
import pandas as pd
from scipy import stats
# Libraries
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from data_cuts import cuts0910, cuts2017, cuts9602
from plotting_functions import filter_rdif, filter_rdif_all
from analyse_functions import Calc_average_Dif_yref,apply_calibration, cal_dif
from constant_variables import *
import warnings
warnings.filterwarnings("ignore")


#plot style variables#
size_label =20
size_title =22

df0910 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_calibrated.csv", low_memory=False)
df17 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_calibrated.csv", low_memory=False)
df17 = df17[df17.iB2 >= 0]
df9602 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_calibrated.csv", low_memory=False)
df17['Year'] = '2017'
df0910['Year'] = '0910'

slist = [0,1,2,3,4,5]
y = [0] * 6

bool_current = True
# year = '0910'
# year_title = '2009/2010'
# pyear = "0910"
# year = '2017'
# year_title = '2017'
# pyear = "2017"
year = 'all'
#
df = df0910
slist = [0,1,3,4]
if year == '2017':
    slist = [0,2,4,5]
    df = df17
    df = df[df.iB2 >= 0 ]
if year == '9602':
    df = df9602


if year == 'all':
    df = pd.concat([df0910, df17], ignore_index=True)
    year_title = '2009/2010/2017'
    pyear = "0910-2017"
    slist = [0, 1, 2,3, 4, 5]

if bool_current:

    ffile = 'current_'

    df['I_OPM_kom_notpc'] = (df['PO3_OPM'] * df['PFcor_kom']) / (df['TPext'] * 0.043085)
    # df['PO3_calc'] = (0.043085 * df['TPext'] * (df['IM'] - df['iB1'])) / (df['PFcor_kom'])
    df['I_OPM'] = (df['PO3_OPM'] * df['PFcor']) / (df['TPext'] * 0.043085)


    df['IminusiB1'] = df['IM'] - df['iB1']
    if year == '2017':df['IminusiB1'] = df['IM'] - df['iB2']
    if year == 'all':
        df.loc[df.Year == '2017','IminusiB1'] = df.loc[df.Year == '2017','IM'] - df.loc[df.Year =='2017','iB2']
        df.loc[df.Year == '0910','IminusiB1'] = df.loc[df.Year == '0910','IM'] - df.loc[df.Year =='0910','iB1']


    # df['ADif'], df['RDif'] = cal_dif(df, 'IM', 'I_OPM_kom', 'ADif', 'RDif')
    df['ADif'], df['RDif'] = cal_dif(df, 'IminusiB1', 'I_OPM_kom_notpc', 'ADif', 'RDif')
    df['ADif_cor'], df['RDif_cor'] = cal_dif(df, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma', 'ADif_cor', 'RDif_cor')
    df['ADif_cal'], df['RDif_cal'] = cal_dif(df, 'I_corrected', 'I_OPM_jma', 'ADif_cal',
                                                               'RDif_cal')

    profl = filter_rdif_all(df)

    # adif_IM, adif_IM_err, rdif_IM, rdif_IM_err, Yp = Calc_average_Dif_yref(profl, 'IM', 'I_OPM_kom','pressure', yref)
    adif_IM, adif_IM_err, rdif_IM, rdif_IM_err, Yp = Calc_average_Dif_yref(profl, 'IminusiB1', 'I_OPM_kom_notpc','pressure', yref)

    adif_IM_deconv10, adif_IM_deconv10_err, rdif_IM_deconv10, rdif_IM_deconv10_err, Yp = \
        Calc_average_Dif_yref(profl, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma','pressure', yref)
    adif_IM_cal, adif_IM_cal_err, rdif_IM_cal, rdif_IM_cal_err, Ypc = \
        Calc_average_Dif_yref(profl, 'I_corrected', 'I_OPM_jma', 'pressure', yref)

if not bool_current:

    ffile = 'pressure_'

    # df['PO3_calc'] = (0.043085 * df['TPext'] * (df['IM'] - df['iB1'])) / (df['PFcor_kom'])

    df['PO3_cal'] = (0.043085 * df['Tpump_cor'] * df['I_corrected']) / (df['PFcor_jma'])


    df['ADif'], df['RDif'] = cal_dif(df, 'PO3_calc', 'PO3_OPM', 'ADif', 'RDif')
    df['ADif_cor'], df['RDif_cor'] = cal_dif(df, 'PO3_cor', 'PO3_OPM', 'ADif_cor', 'RDif_cor')
    df['ADif_cal'], df['RDif_cal'] = cal_dif(df, 'PO3_cal', 'PO3_OPM', 'ADif_cal',
                                             'RDif_cal')
    profl = filter_rdif_all(df)

    adif_IM, adif_IM_err, rdif_IM, rdif_IM_err, Yp = Calc_average_Dif_yref(profl, 'PO3_calc', 'PO3_OPM', 'pressure', yref)
    adif_IM_deconv10, adif_IM_deconv10_err, rdif_IM_deconv10, rdif_IM_deconv10_err, Yp = \
        Calc_average_Dif_yref(profl, 'PO3_cor', 'PO3_OPM', 'pressure', yref)
    adif_IM_cal, adif_IM_cal_err, rdif_IM_cal, rdif_IM_cal_err, Ypc = \
        Calc_average_Dif_yref(profl, 'PO3_cal', 'PO3_OPM', 'pressure', yref)


nsim = [0] * 6
a_year = [0] * 6
b_year = [0] * 6

pcut = 13
urdif = 30
lrdif = -30
sign = ['+'] * len(profl)
labelc = [0] * len(profl)

#all in one plot original, convolutued, calibrated

print('slit', slist)

for k in slist:

    profl[k]['pair_nan'] = 0
    profl[k].loc[profl[k].Pair.isnull(), 'pair_nan'] = 1

    dft = profl[k][profl[k].pair_nan == 0]
    dft = dft[(dft.RDif_cor < urdif) & (dft.RDif_cor > lrdif)]
    filt_p = dft.Pair >= pcut
    y[k] = np.array(dft[filt_p]['Pair'])
    labelc[k] = b[k]
    if pyear == '0910':
        a_year[k] = a_0910[k]
        b_year[k] = b_0910[k]
        labelc[k] = b_0910[k]

    if pyear == '2017':
        a_year[k] = a_2017[k]
        b_year[k] = b_2017[k]
        labelc[k] = b_2017[k]
    if  pyear == "0910-2017":
        a_year[k] = a[k]
        b_year[k] = b[k]
        labelc[k] = b[k]


    if b_year[k] < 0:
        sign[k] = '-'
        labelc[k] = -1 * b_year[k]
    # if b[k] < 0:
    #     sign[k] = '-'
    #     labelc[k] = -1 * b[k]

    plotname = f'{pyear}_{labellist[k]}_Scatter_RDif_less30'

    nsim[k] = len(profl[k].drop_duplicates(['Sim', 'Team']))

    maintitle = 'JOSIE ' + year_title + " " + labellist[k]
    ytitle = 'Pressure [hPa]'
    xtitle = 'Current'
    xrtitle = 'Relative Difference [%] \n (Sonde - OPM)/OPM'

    fig = plt.figure(figsize=(15, 12))
    # plt.suptitle("GridSpec Inside GridSpec")
    plt.suptitle(maintitle, fontsize=size_title, y = 0.93)

    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 2])
    gs.update(wspace=0.0005, hspace=0.05)
    ax0 = plt.subplot(gs[0])
    plt.yscale('log')
    plt.ylim([1000, 5])
    plt.xlim([-40, 40])
    ax0.set_xticklabels([-40, -30, -20, -10, 0, 10, 20 ,30])
    plt.yticks(fontsize=size_label)
    plt.xticks(fontsize=size_label)

    ax0.yaxis.set_major_formatter(ScalarFormatter())
    plt.grid(True)
    plt.xlabel(xrtitle, fontsize=size_label)
    plt.ylabel(ytitle, fontsize=size_label)
    ax0.tick_params(which='major', width=2)
    ax0.tick_params(which='minor',width=2)

    plt.plot(profl[k]['RDif'], profl[k]['Pair'], color=cbl[k], linestyle='None', marker='o', markersize=0.5,
             label = 'Conventional')
    ax0.errorbar(rdif_IM[k], Yp, xerr=rdif_IM_err[k], color='black', linewidth=3, elinewidth=1, capsize=1, capthick=1)
    ax0.plot([], [], ' ', label=f"nsim={nsim[k]}")

    # handles, labels = ax0.get_legend_handles_labels()
    # order = [0, 2, 1]
    # ax0.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
    #            loc='upper left', frameon=True, fontsize='x-large', markerscale=7 )
    # ax0.legend(loc='lower left', frameon=True, fontsize='xx-large', markerscale=10, handletextpad=0.05)
    ax0.legend(loc='best', frameon=True, fontsize='xx-large', markerscale=10,  handletextpad=0.1)

    ax0.axvline(x=0, color='grey', linestyle='--', linewidth=3)

    # 2nd grid TRRM
    ax1 = plt.subplot(gs[1])
    plt.yscale('log')
    plt.ylim([1000, 5])
    plt.xlim([-40, 40])
    plt.xlabel(xrtitle, fontsize=size_label)
    plt.xticks(fontsize=size_label)

    ax1.tick_params(which='major', width=2)
    ax1.tick_params(which='minor',width=2)
    plt.plot(profl[k]['RDif_cor'], profl[k]['Pair'], color=cbl[k], linestyle='None', marker='o',
             label=f'TRRM', markersize=0.5)


    plt.plot(a_year[k] + b_year[k] * np.log10(y[k]), y[k], color='black', linestyle=':', linewidth=6,
             label=rf'{round(a_year[k], 2)} {sign[k]} {round(labelc[k], 2)}$\cdot$log(P)')
    ax1.errorbar(rdif_IM_deconv10[k], Yp, xerr=rdif_IM_deconv10_err[k],
                 color='black', linewidth=3, elinewidth=1, capsize=1, capthick=1)

    ax1.set_yticklabels([])
    ax1.legend(loc='best', frameon=True, fontsize='xx-large', markerscale=10, handletextpad=0.1 )

    # ax1.legend(loc='lower left', frameon=True, fontsize='xx-large', markerscale=10, handletextpad=0.1)
    ax1.axvline(x=0, color='grey', linestyle='--', linewidth=3)
    plt.grid(True)

    plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v2/png/{ffile}{plotname}.png')
    # plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v2/eps/{ffile}{plotname}.eps')
    # plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v2/pdf/{ffile}{plotname}.pdf')
    plt.show()

    plt.close()



# PLOT ALL in ONE

for k in slist:

    a[k] = a[k]
    plotname = f'{pyear}_{labellist[k]}_Triple_RDif_less30'


    nsim[k] = len(profl[k].drop_duplicates(['Sim', 'Team']))

    maintitle = 'JOSIE ' + year + " " + labellist[k]
    ytitle = 'Pressure [hPa]'
    xtitle = 'Current'
    xrtitle = 'Relative Difference [%] \n (Sonde - OPM)/OPM'

    # fig, ax = plt.figure(figsize=(6, 12))
    # fig, ax = plt.subplots(figsize=(8, 12), layout = 'constrained')
    fig, ax = plt.subplots(figsize=(8, 12))
    # ,
    # plt.suptitle("GridSpec Inside GridSpec")
    plt.suptitle(maintitle, fontsize=size_title, y = 0.93)


    # ax2 = plt.subplot(gs[2])
    plt.yscale('log')
    plt.ylim([1000, 5])
    # plt.xlim([-15,15])
    plt.xlim([-40, 40])
    plt.yticks(fontsize=size_label)
    plt.xticks(fontsize=size_label)
    ax.tick_params(which='major', width=2)
    ax.tick_params(which='minor',width=2)

    ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.grid(True)
    plt.xlabel(xrtitle, fontsize=size_label)

    ax.errorbar(rdif_IM[k], Yp, xerr=rdif_IM_err[k], label=f'Conventional',
                 color=cbl[0], linewidth=2, elinewidth=1, capsize=1, capthick=1)

    ax.errorbar(rdif_IM_deconv10[k], Yp, xerr=rdif_IM_deconv10_err[k], label=f'TRRM',
                 color=cbl[4], linewidth=2, elinewidth=1, capsize=1, capthick=1)

    ax.errorbar(rdif_IM_cal[k], Yp, xerr=rdif_IM_cal_err[k], label=f'TRRM + Calibration',
                 color=cbl[3], linewidth=2, elinewidth=1, capsize=1, capthick=1)

    # ax.legend(loc='upper left', frameon=True, fontsize='xx-large', markerscale=3)
    ax.legend(loc='best', frameon=True, fontsize='xx-large', markerscale=3)

    ax.axvline(x=0, color='grey', linestyle='--', linewidth=3)
    plt.grid(True)

    # plt.tight_layout()

    # plt.savefig('grid_figure.pdf')
    plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v2/png/{ffile}{plotname}.png')
    # plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v2/eps/{ffile}{plotname}.eps')
    # plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v2/pdf/{ffile}{plotname}.pdf')
    # plt.show()

    plt.close()
##################################################################################################################


