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

df0910 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_calibrated.csv", low_memory=False)
df17 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_calibrated.csv", low_memory=False)
df9602 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_calibrated.csv", low_memory=False)

slist = [0,1,2,3,4,5]
y = [0] * 6

# for i in slist:
#     nsim[i] = len(prof[i].drop_duplicates(['Sim', 'Team']))



year = '0910'
df = df0910
slist = [0,1,3,4]
if year == '2017':
    slist = [0,2,4,5]
    df = df17
if year == '9602':
    df = df9602

df['ADif'], df['RDif'] = cal_dif(df, 'IM', 'I_OPM_kom', 'ADif', 'RDif')
df['ADif_cor'], df['RDif_cor'] = cal_dif(df, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma', 'ADif_cor', 'RDif_cor')
df['ADif_cal'], df['RDif_cal'] = cal_dif(df, 'I_corrected', 'I_OPM_jma', 'ADif_cal',
                                                           'RDif_cal')

profl = filter_rdif_all(df)

adif_IM, adif_IM_err, rdif_IM, rdif_IM_err, Yp = Calc_average_Dif_yref(profl, 'IM', 'I_OPM_kom','pressure', yref)
adif_IM_deconv10, adif_IM_deconv10_err, rdif_IM_deconv10, rdif_IM_deconv10_err, Yp = \
    Calc_average_Dif_yref(profl, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma','pressure', yref)
adif_IM_cal, adif_IM_cal_err, rdif_IM_cal, rdif_IM_cal_err, Ypc = \
    Calc_average_Dif_yref(profl, 'I_corrected', 'I_OPM_jma', 'pressure', yref)




nsim = [0] * 6
pcut = 13
urdif = 30
lrdif = -30
sign = ['+'] * len(profl)
labelc = [0] * len(profl)

#all in one plot original, convolutued, calibrated
year = '2009/2010'
pyear = "0910"
for k in slist:

    profl[k]['pair_nan'] = 0
    profl[k].loc[profl[k].Pair.isnull(), 'pair_nan'] = 1

    dft = profl[k][profl[k].pair_nan == 0]
    dft = dft[(dft.RDif_cor < urdif) & (dft.RDif_cor > lrdif)]
    filt_p = dft.Pair >= pcut
    y[k] = np.array(dft[filt_p]['Pair'])
    labelc[k] = a[k]

    if b[k] < 0:
        sign[k] = '-'
        labelc[k] = -1 * b[k]

    plotname = f'{pyear}_{labellist[k]}_Scatter_RDif_less30'

    nsim[k] = len(profl[k].drop_duplicates(['Sim', 'Team']))

    maintitle = 'JOSIE ' + year + " " + labellist[k]
    ytitle = 'Pressure [hPa]'
    xtitle = 'Current'
    xrtitle = 'Relative Difference [%] \n (Sonde - OPM)/OPM'

    fig = plt.figure(figsize=(15, 12))
    # plt.suptitle("GridSpec Inside GridSpec")
    plt.suptitle(maintitle, fontsize=18, y = 0.95)

    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 2])
    gs.update(wspace=0.0005, hspace=0.05)
    ax0 = plt.subplot(gs[0])
    plt.yscale('log')
    plt.ylim([1000, 5])
    plt.xlim([-40, 40])
    ax0.set_xticklabels([-40, -30, -20, -10, 0, 10, 20 ,30])
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)

    ax0.yaxis.set_major_formatter(ScalarFormatter())
    plt.grid(True)
    plt.xlabel(xrtitle, fontsize=16)
    plt.ylabel(ytitle, fontsize=16)
    ax0.tick_params(which='major', width=2)
    ax0.tick_params(which='minor',width=2)

    plt.plot(profl[k]['RDif'], profl[k]['Pair'], color=cbl[k], linestyle='None', marker='o', markersize=0.5,
             label = 'Conventional Method')
    ax0.errorbar(rdif_IM[k], Yp, xerr=rdif_IM_err[k], color='black', linewidth=3, elinewidth=1, capsize=1, capthick=1)
    ax0.plot([], [], ' ', label=f"nsim={nsim[k]}")

    handles, labels = ax0.get_legend_handles_labels()
    # order = [0, 2, 1]
    # ax0.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
    #            loc='upper left', frameon=True, fontsize='x-large', markerscale=7 )
    ax0.legend(loc='lower left', frameon=True, fontsize='x-large', markerscale=10, handletextpad=0.1)
    ax0.axvline(x=0, color='grey', linestyle='--', linewidth=3)

    # 2nd grid TRRM
    ax1 = plt.subplot(gs[1])
    plt.yscale('log')
    plt.ylim([1000, 5])
    plt.xlim([-40, 40])
    plt.xlabel(xrtitle, fontsize=16)
    plt.xticks(fontsize=16)

    ax1.tick_params(which='major', width=2)
    ax1.tick_params(which='minor',width=2)
    plt.plot(profl[k]['RDif_cor'], profl[k]['Pair'], color=cbl[k], linestyle='None', marker='o',
             label=f'TRRM', markersize=0.5)

    # plt.plot(coeff[k][1] + coeff[k][0] * np.log10(y[k]), y[k], color='black', linestyle=':', linewidth=4,
    #          label=rf'{round(coeff[k][1], 3)}{sign[k]}{round(labelc[k], 3)}$\cdot$log(P)')
    plt.plot(a[k] + b[k] * np.log10(y[k]), y[k], color='black', linestyle=':', linewidth=6,
             label=rf'{round(a[k], 2)} {sign[k]} {round(labelc[k], 2)}$\cdot$log(P)')
    ax1.errorbar(rdif_IM_deconv10[k], Yp, xerr=rdif_IM_deconv10_err[k],
                 color='black', linewidth=3, elinewidth=1, capsize=1, capthick=1)

    ax1.set_yticklabels([])

    ax1.legend(loc='lower left', frameon=True, fontsize='x-large', markerscale=10, handletextpad=0.1)
    ax1.axvline(x=0, color='grey', linestyle='--')
    plt.grid(True)


    #
    plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/png/v1_' + plotname + '.png')
    plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/eps/v1_' + plotname + '.eps')
    plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/pdf/v1_' + plotname + '.pdf')

    plt.show()



# PLOT ALL in ONE

for k in slist:

    s = k
    plotname =  f'yref5_OPM_updated_{pyear}_{labellist[s]}_AllinOne_RDif_less30_'

    nsim[k] = len(profl[k].drop_duplicates(['Sim', 'Team']))

    maintitle = 'JOSIE ' + year + " " + labellist[s]
    ytitle = 'Pressure [hPa]'
    xtitle = 'Current'
    xrtitle = 'Relative Difference [%] \n (Sonde - OPM)/OPM'

    # fig, ax = plt.figure(figsize=(6, 12))
    fig, ax = plt.subplots(figsize=(8, 12), layout = 'constrained')
    # ,
    # plt.suptitle("GridSpec Inside GridSpec")
    plt.suptitle(maintitle, fontsize=16)

    # ax2 = plt.subplot(gs[2])
    plt.yscale('log')
    plt.ylim([1000, 5])
    # plt.xlim([-15,15])
    plt.xlim([-40, 40])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.grid(True)
    plt.xlabel(xrtitle, fontsize=16)

    ax.errorbar(rdif_IM[k], Yp, xerr=rdif_IM_err[k], label=f'Averaged Relative Difference',
                 color=cbl[0], linewidth=2, elinewidth=1, capsize=1, capthick=1)

    ax.errorbar(rdif_IM_deconv10[k], Yp, xerr=rdif_IM_deconv10_err[k], label=f'Averaged Relative Difference - New Method',
                 color=cbl[1], linewidth=2, elinewidth=1, capsize=1, capthick=1)

    ax.errorbar(rdif_IM_cal[k], Yp, xerr=rdif_IM_cal_err[k], label=f'Averaged Relative Difference - Calibrated',
                 color=cbl[3], linewidth=2, elinewidth=1, capsize=1, capthick=1)

    ax.legend(loc='upper left', frameon=True, fontsize='large', markerscale=3)
    ax.axvline(x=0, color='grey', linestyle='--')
    plt.grid(True)

    # plt.tight_layout()

    # plt.savefig('grid_figure.pdf')
    plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/png/upd_' + plotname + '.png')
    plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/eps/upd_' + plotname + '.eps')
    plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/pdf/upd_' + plotname + '.pdf')
    # plt.show()


##################################################################################################################


