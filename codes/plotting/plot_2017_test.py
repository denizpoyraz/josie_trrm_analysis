import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter

from analyse_functions import calc_average_df_pressure, set_columns_nopair_dependence
from constant_variables import *
from data_cuts import cuts0910, cuts2017
from homogenization_functions import return_phipcor
from plotting_functions import filter_rdif_all
from plotting_functions import filter_rdif, filter_rdif_all
from analyse_functions import Calc_average_Dif_yref,apply_calibration, cal_dif
from constant_variables import *
from matplotlib.ticker import ScalarFormatter, MultipleLocator


df17 = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_calibrated_sm_hv.csv", low_memory=False)
df17 = df17[df17.iB2 >= 0]

df17_2b = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_deconv_2023paper_ib2_sm_hv_2beta.csv", low_memory=False)
df17_2b = df17_2b[df17_2b.iB2 >= 0]
df17 = cuts2017(df17)
df17_2b = cuts2017(df17_2b)

slist = [0,2,4,5]

year = '2017'

# df17['PO3_cal'] = (0.043085 * df17['Tpump_cor'] * df17['I_corrected']) / (df17['PFcor_jma'])
df17['PO3_islow'] = (0.043085 * df17['Tpump_cor'] * df17['Islow_conv']) / (df17['PFcor_jma'])
df17['PO3_iB0'] = (0.043085 * df17['Tpump_cor'] * df17['iB0']) / (df17['PFcor_jma'])
df17['PO3_iB1'] = (0.043085 * df17['Tpump_cor'] * df17['iB1']) / (df17['PFcor_jma'])

df17['ADif'], df17['RDif'] = cal_dif(df17, 'PO3_dqa', 'PO3_OPM', 'ADif', 'RDif')
df17['ADif_cor'], df17['RDif_cor'] = cal_dif(df17, 'PO3_cor', 'PO3_OPM', 'ADif_cor', 'RDif_cor')

profl = filter_rdif_all(df17)

adif_IM, adif_IM_err, rdif_IM, rdif_IM_err, Yp = Calc_average_Dif_yref(profl, 'PO3_dqa', 'PO3_OPM', 'pressure', yref)
adif_IM_deconv10, adif_IM_deconv10_err, rdif_IM_deconv10, rdif_IM_deconv10_err, Yp = \
    Calc_average_Dif_yref(profl, 'PO3_cor', 'PO3_OPM', 'pressure', yref)


##for the 2*beta deconv. 2017 sample
# df17_2b['PO3_cal'] = (0.043085 * df17_2b['Tpump_cor'] * df17_2b['I_corrected']) / (df17_2b['PFcor_jma'])
df17_2b['PO3_cor'] = (0.043085 * df17_2b['Tpump_cor'] * df17_2b['Ifast_minib0_deconv_sm10']) / (df17_2b['PFcor_jma'])

df17_2b['PO3_islow'] = (0.043085 * df17_2b['Tpump_cor'] * df17_2b['Islow_conv']) / (df17_2b['PFcor_jma'])
df17_2b['PO3_iB0'] = (0.043085 * df17_2b['Tpump_cor'] * df17_2b['iB0']) / (df17_2b['PFcor_jma'])
df17_2b['PO3_iB1'] = (0.043085 * df17_2b['Tpump_cor'] * df17_2b['iB1']) / (df17_2b['PFcor_jma'])

df17_2b['ADif'], df17_2b['RDif'] = cal_dif(df17_2b, 'PO3_dqa', 'PO3_OPM', 'ADif', 'RDif')
df17_2b['ADif_cor'], df17_2b['RDif_cor'] = cal_dif(df17_2b, 'PO3_cor', 'PO3_OPM', 'ADif_cor', 'RDif_cor')

profl_2b = filter_rdif_all(df17_2b)

adif_2b_IM, adif_2b_IM_err, rdif_2b_IM, rdif_2b_IM_err, Yp = Calc_average_Dif_yref(profl_2b, 'PO3_dqa', 'PO3_OPM', 'pressure', yref)
adif_2b_IM_deconv10, adif_2b_IM_deconv10_err, rdif_2b_IM_deconv10, rdif_2b_IM_deconv10_err, Yp = \
    Calc_average_Dif_yref(profl_2b, 'PO3_cor', 'PO3_OPM', 'pressure', yref)




size_label =14
size_title =16
nsim = [0] * 6



column_list = ['Pair','Tsim','Ifast_minib0_deconv_sm10','Islow_conv','TPext', 'Tpump_cor',
               'PO3','PO3_dqa', 'PO3_cor', 'PO3_islow', 'PO3_OPM','PO3_iB0', 'PO3_iB1']

# yrefd = [1000, 850, 700, 550, 400, 350, 300, 200, 150, 100, 75, 50, 35,30, 25, 15,9, 8, 6]


#v1
for k in slist:

    dfi = profl[k]
    df = calc_average_df_pressure(dfi, column_list, yref)
    nop_columns = ['PFcor', 'iB0', 'iB1', 'iB2']
    df = set_columns_nopair_dependence(dfi, df,  nop_columns)

    dfi_2b = profl_2b[k]
    df_2b = calc_average_df_pressure(dfi_2b, column_list, yref)
    nop_columns = ['PFcor', 'iB0', 'iB1', 'iB2']
    df_2b = set_columns_nopair_dependence(dfi_2b, df_2b, nop_columns)

    # plotname = f'v2_ADif_{pre1}{pyear}_{labellist[k]}_Scatter'
    #
    nsim[k] = len(profl[k].drop_duplicates(['Sim', 'Team']))

    maintitle = 'JOSIE ' + year + " " + labellist[k]
    ytitle = 'Pressure [hPa]'
    # xtitle = 'Current'
    xrtitle = 'Sonde - OPM [mPa]'

    xptitle = 'Ozone Partial Pressure [mPa]'
    xreltitle = 'Sonde - OPM [%]'

    fig = plt.figure(figsize=(15, 12))
    # plt.suptitle("GridSpec Inside GridSpec")
    plt.suptitle(maintitle, fontsize=size_title, y=0.93)

    # gs = gridspec.GridSpec(1, 2, width_ratios=[2, 2])
    gs = gridspec.GridSpec(1, 4, width_ratios=[2, 2, 2, 2])

    gs.update(wspace=0.0005, hspace=0.05)
    ax0 = plt.subplot(gs[0])
    plt.yscale('log')
    plt.ylim([1000, 5])
    # plt.xlim([-3, 3])
    # plt.xticks([-2, -1, 0, 1, 2])

    # ax0.set_xticklabels(['',-2,-1,0,1,2,''])
    plt.yticks(fontsize=size_label)
    plt.xticks(fontsize=size_label)

    ax0.yaxis.set_major_formatter(ScalarFormatter())

    plt.grid(True)
    plt.xlabel(xptitle, fontsize=size_label)
    plt.ylabel(ytitle, fontsize=size_label)
    ax0.tick_params(which='major', width=3)
    ax0.tick_params(which='minor', width=3)
    # ax0.xaxis.set_minor_locator(MultipleLocator(2))
    # ax0.xaxis.set_tick_params(length=5, which='minor')
    ax0.xaxis.set_tick_params(length=10, which='major')
    ax0.yaxis.set_tick_params(length=5, which='minor')
    ax0.yaxis.set_tick_params(length=10, which='major')


    # ax0.errorbar(df['PO3_cor'], Yp, xerr=adif_IM_err[k], color='black', linewidth=3, elinewidth=1, capsize=1, capthick=1)
    print(df['PO3_cor'])
    # plt.plot(np.array(df['PO3_cor']), np.array(df['Pair']), label=f"TRRM", color = cbl[k])
    plt.plot(np.array(df['PO3_dqa']), np.array(df['Pair']), label=f"Conventional", color = 'black')

    plt.plot(np.array(df['PO3_cor']), np.array(df['Pair']), label=f"TRRM", color = cbl[k])
    plt.plot(np.array(df_2b['PO3_cor']), np.array(df_2b['Pair']), label=f"TRRM 2*beta", color = 'skyblue')
    plt.plot(np.array(df_2b['PO3_OPM']), np.array(df_2b['Pair']), label=f"OPM", color = 'darkgreen')


    ax0.legend(loc='best', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1)

    # ax0.axvline(x=0, color='grey', linestyle='--', linewidth=3)

    # 2nd grid TRRM
    ax1 = plt.subplot(gs[1])
    plt.yscale('log')
    plt.ylim([1000, 5])
    # plt.xlim([-3, 3])
    # # ax1.set_xticklabels(['',-2,-1,0,1,2,''])
    # plt.xticks([-2, -1, 0, 1, 2])

    plt.xlabel(xptitle, fontsize=size_label)
    plt.xticks(fontsize=size_label)

    ax1.tick_params(which='major', width=3)
    ax1.tick_params(which='minor', width=3)
    ax1.xaxis.set_minor_locator(MultipleLocator(2))
    ax1.xaxis.set_tick_params(length=5, which='minor')
    ax1.xaxis.set_tick_params(length=10, which='major')
    ax1.yaxis.set_tick_params(length=5, which='minor')
    ax1.yaxis.set_tick_params(length=10, which='major')
    # plt.plot(profl[k]['ADif_cor'], profl[k]['Pair'], color=cbl[k], linestyle='None', marker='o',
    #          label=f'TRRM', markersize=0.5)
    plt.plot(df['PO3_islow'], df['Pair'], label=r"PO3$_{I_S}$", color='goldenrod', linewidth=2)

    plt.plot(df_2b['PO3_islow'], df_2b['Pair'], label=r"PO3$_{I_S}$, 2*beta", color='yellowgreen', linewidth=2)
    plt.plot(-1 * df['PO3_iB0'], df['Pair'], label=r"-1*PO3$_{iB0}$", color='darkcyan', linewidth=2)
    plt.plot(-1 * df['PO3_iB1'], df['Pair'], label=r"-1*PO3$_{iB1}$", color='crimson', linewidth=2)


    ax1.set_yticklabels([])
    ax1.legend(loc='best', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1)

    # ax1.legend(loc='lower left', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1)
    # ax1.axvline(x=0, color='grey', linestyle='--', linewidth=3)
    plt.grid(True)

    # # trrm + calibrated#
    ax2 = plt.subplot(gs[2])
    plt.yscale('log')
    plt.ylim([1000, 5])
    # plt.xlim([-3, 3])
    # # ax2.set_xticklabels(['',-2,-1,0,1,2,''])
    # plt.xticks([-2, -1, 0, 1, 2])
    #
    plt.xlabel(xrtitle, fontsize=size_label)
    plt.xticks(fontsize=size_label)
    #
    ax2.tick_params(which='major', width=3)
    ax2.tick_params(which='minor', width=3)
    ax2.xaxis.set_minor_locator(MultipleLocator(2))
    ax2.xaxis.set_tick_params(length=5, which='minor')
    ax2.xaxis.set_tick_params(length=10, which='major')
    ax2.yaxis.set_tick_params(length=5, which='minor')
    ax2.yaxis.set_tick_params(length=10, which='major')
    plt.plot(adif_IM_deconv10[k], Yp, color=cbl[k], label=f'TRRM', linewidth=3)
    plt.plot(adif_2b_IM_deconv10[k], Yp, color='skyblue', label=f'TRRM, 2*beta', linewidth=3)

    plt.plot(adif_IM[k], Yp, color='black', label=f'Conventional', linewidth=3)

    ax2.set_yticklabels([])
    ax2.legend(loc='best', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1)

    # ax2.legend(loc='lower left', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1)
    ax2.axvline(x=0, color='grey', linestyle='--', linewidth=3)
    plt.grid(True)

    ax3 = plt.subplot(gs[3])
    plt.yscale('log')
    plt.ylim([1000, 5])
    # plt.xlim([-3, 3])
    # # ax3.set_xticklabels(['',-2,-1,0,1,2,''])
    # plt.xticks([-2, -1, 0, 1, 2])
    #
    plt.xlabel(xreltitle, fontsize=size_label)
    plt.xticks(fontsize=size_label)
    #
    ax3.tick_params(which='major', width=3)
    ax3.tick_params(which='minor', width=3)
    ax3.xaxis.set_minor_locator(MultipleLocator(2))
    ax3.xaxis.set_tick_params(length=5, which='minor')
    ax3.xaxis.set_tick_params(length=10, which='major')
    ax3.yaxis.set_tick_params(length=5, which='minor')
    ax3.yaxis.set_tick_params(length=10, which='major')
    plt.plot(rdif_IM_deconv10[k], Yp, color=cbl[k], label=f'TRRM', linewidth=3)
    plt.plot(rdif_2b_IM_deconv10[k], Yp, color='skyblue', label=f'TRRM, 2*beta', linewidth=3)

    plt.plot(rdif_IM[k], Yp, color='black', label=f'Conventional', linewidth=3)

    ax3.set_yticklabels([])
    ax3.legend(loc='best', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1)

    # ax3.legend(loc='lower left', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1)
    ax3.axvline(x=0, color='grey', linestyle='--', linewidth=3)
    plt.grid(True)

    # plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/checks/v2_{year}_{labellist[k]}.png')

    plt.show()

    plt.close()


# df0910t = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_calibrated_sm_hv.csv", low_memory=False)
#
# df17 = df0910t.drop(['Unnamed: 0.1', 'index', 'Tact', 'Tair', 'Tinlet', 'TPint', 'I_Pump', 'VMRO3', 'VMRO3_OPM',
#  'ADif_PO3S', 'RDif_PO3S','Z', 'Header_Team', 'Header_Sim', 'Header_PFunc', 'Header_PFcor',
#  'Header_IB1', 'Simulator_RunNr', 'Date', 'Ini_Prep_Date', 'Prep_SOP', 'SerialNr', 'SerNr', 'Date_1',
#  'SondeAge', 'Solutions', 'Volume_Cathode', 'ByPass_Cell', 'Current_10min_after_noO3', 'Resp_Time_4_1p5_sec',
# 'RespTime_1minOver2min', 'Final_BG', 'T100', 'mlOvermin', 'T100_post', 'mlOverminp1', 'RespTime_4_1p5_sec_p1','RespTime_1minOver2min_microamps', 'PostTestSolution_Lost_gr', 'PumpMotorCurrent', 'PumpMotorCurrent_Post',
#  'PF_Unc', 'PF_Cor', 'BG', 'plog', 'Tcell', 'TcellC','Pw', 'massloss', 'Tboil', 'total_massloss', 'I_conv_slow',
#                       'I_conv_slow_komhyr'], axis=1)
#
# slist = [0,1,3,4]
# year = '2009-2010'