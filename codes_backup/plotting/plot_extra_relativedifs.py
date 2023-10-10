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

year = '2017'
# year = '0910'

df17 = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_calibrated_sm_hv.csv", low_memory=False)
# df17 = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_calibrated_sm_hv.csv", low_memory=False)
df17 = df17[df17.iB2 >= 0]
df17 = df17[df17.iB0 >= -0.4]
df17 = df17[df17.iB0 < 0.1]

df17 = cuts2017(df17)

slist = [0,2,4,5]

if year == '0910':
    slist = [0,1,3,4]
    df17 = cuts0910(df17)

if year == '0910':
    # df17['ADif_islow'], df17['RDif_islow'] = cal_dif(df17, 'Islow_conv_ib1_decay', 'IM', 'ADif_islow', 'RDif_islow')
    # df17['ADif_ifdc'], df17['RDif_ifdc'] = cal_dif(df17, 'Ifast_minib0_deconv_ib1_decay', 'IM', 'ADif_ifdc', 'RDif_ifdc')
    # df17['ratio_islow'] = df17['Islow_conv_ib1_decay']/df17['IM']
    # df17['ratio_ifdc'] = df17['Ifast_minib0_deconv_ib1_decay']/df17['IM']
    df17['Islow_conv'] = df17['I_slow_conv_ib1_decay']
    df17['Ifast_minib0_deconv'] = df17['Ifast_minib0_deconv_ib1_decay']
    df17['Ifast_minib0'] = df17['Ifast_minib0_ib1_decay']

df17['IM-iB0'] = df17['IM'] - df17['iB0']

df17['ADif_ib0'], df17['RDif_ib0'] = cal_dif(df17, 'iB0', 'IM', 'ADif_ib0', 'RDif_ib0')
df17['ADif_islow'], df17['RDif_islow'] = cal_dif(df17, 'Islow_conv', 'IM', 'ADif_islow', 'RDif_islow')
df17['ADif_ifdc'], df17['RDif_ifdc'] = cal_dif(df17, 'Ifast_minib0_deconv', 'IM', 'ADif_ifdc', 'RDif_ifdc')
df17['ADif_ifdc_noib0'], df17['RDif_ifdc_noib0'] = cal_dif(df17, 'Ifast_deconv', 'IM', 'ADif_ifdc_noib0', 'RDif_ifdc_noib0')

df17['ADif_ib0_t'], df17['RDif_ib0_t'] = cal_dif(df17, 'iB0', 'Ifast_minib0_deconv', 'ADif_ib0_t', 'RDif_ib0_t')
df17['ADif_islow_t'], df17['RDif_islow_t'] = cal_dif(df17, 'Islow_conv', 'Ifast_minib0_deconv', 'ADif_islow_t', 'RDif_islow_t')
df17['ADif_if_t'], df17['RDif_if_t'] = cal_dif(df17, 'Ifast', 'Ifast_deconv', 'ADif_if_t', 'RDif_if_t')
df17['ADif_ifm_t'], df17['RDif_ifm_t'] = cal_dif(df17, 'Ifast_minib0', 'Ifast_minib0_deconv', 'ADif_ifm_t', 'RDif_ifm_t')


df17['ratio_ib0'] = df17['iB0']/df17['IM']
df17['ratio_islow'] = df17['Islow_conv']/df17['IM']
df17['ratio_ifdc'] = df17['Ifast_minib0_deconv']/df17['IM']
df17['ratio_ifdc_noib0'] = df17['Ifast_deconv']/df17['IM']

df17['ratio_2'] = df17['iB0']/df17['IM']
df17['ratio_3'] = df17['Islow_conv']/df17['IM']
df17['ratio_6'] = (df17['Ifast_minib0_deconv'] - df17['Ifast_minib0'])/df17['IM']
df17['ratio_6d'] = df17['Ifast_minib0_deconv']/df17['IM']
df17['ratio_6b'] = df17['Ifast_minib0']/df17['IM']

df17['ratio_2dc'] = df17['iB0']/df17['Ifast_minib0_deconv']
df17['ratio_3dc'] = df17['Islow_conv']/df17['Ifast_minib0_deconv']
df17['ratio_6dc'] = (df17['Ifast_minib0_deconv'] - df17['Ifast_minib0'])/df17['Ifast_minib0_deconv']
df17['ratio_6ddc'] = df17['IM']/df17['Ifast_minib0_deconv']
df17['ratio_6bdc'] = df17['Ifast_minib0']/df17['Ifast_minib0_deconv']

df17['ratio_7'] = (df17['PFcor_kom'] - df17['PFcor_jma'])/df17['PFcor_jma']
df17['ratio_8'] = df17['ratio_2'] + df17['ratio_3'] + df17['ratio_6'] + df17['ratio_7']
df17['ratio_8d'] = df17['ratio_2'] + df17['ratio_3'] + df17['ratio_6d']
df17['ratio_8b'] = df17['ratio_2'] + df17['ratio_3'] + df17['ratio_6b']

df17['ratio_8dc'] = df17['ratio_2dc'] + df17['ratio_3dc'] + df17['ratio_6ddc']
df17['ratio_8ddc'] = df17['ratio_2dc'] + df17['ratio_3dc'] + df17['ratio_6ddc']
df17['ratio_8bdc'] = df17['ratio_2'] + df17['ratio_3dc'] + df17['ratio_6bdc']

profl = filter_rdif_all(df17)

column_list = ['Pair','Tsim','Islow_conv','TPext', 'Tpump_cor','Ifast_minib0_deconv',
               'Ifast_deconv','IM','iB0', 'I_OPM_jma','ratio_ib0','ratio_islow','ratio_ifdc',
               'RDif_ib0', 'RDif_islow', 'RDif_ifdc', 'RDif_ifdc_noib0','ratio_ifdc_noib0',
               'RDif_ib0_t','RDif_islow_t','RDif_if_t', 'RDif_ifm_t', 'IM-iB0', 'ratio_2', 'ratio_3', 'ratio_6',
               'ratio_7', 'ratio_8', 'ratio_6d', 'ratio_8d', 'ratio_6b', 'ratio_8b',
               'ratio_2dc', 'ratio_3dc', 'ratio_6dc', 'ratio_6ddc','ratio_8dc', 'ratio_8ddc', 'ratio_6bdc', 'ratio_8bdc']


size_label =12
size_title =16
nsim = [0] * 6


#
# column_list = ['Pair','Tsim','Ifast_minib0_deconv_sm10','Islow_conv','TPext', 'Tpump_cor',
#                'PO3','PO3_dqa', 'PO3_cor', 'PO3_islow', 'PO3_OPM','PO3_iB0', 'PO3_iB1']

# yrefd = [1000, 850, 700, 550, 400, 350, 300, 200, 150, 100, 75, 50, 35,30, 25, 15,9, 8, 6]


#v1
for k in slist:

    dfi = profl[k]
    df = calc_average_df_pressure(dfi, column_list, yref)
    nop_columns = ['PFcor','iB0',  'iB1', 'iB2']
    df = set_columns_nopair_dependence(dfi, df,  nop_columns)


    # plotname = f'v2_ADif_{pre1}{pyear}_{labellist[k]}_Scatter'
    #
    nsim[k] = len(profl[k].drop_duplicates(['Sim', 'Team']))

    maintitle = 'JOSIE ' + year + " " + labellist[k]
    ytitle = 'Pressure [hPa]'
    xtitle = r'Current [\muA]'
    # xrtitle = 'Sonde - OPM [mPa]'
    xrtitle = r'Sonde - OPM [\muA]'


    xptitle = 'Ozone Partial Pressure [\muA]]'
    xreltitle = 'Sonde - OPM [%]'

    fig = plt.figure(figsize=(15, 12))
    # plt.suptitle("GridSpec Inside GridSpec")
    plt.suptitle(maintitle, fontsize=size_title, y=0.93)

    # gs = gridspec.GridSpec(1, 2, width_ratios=[2, 2])
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 2])

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

    # plt.plot(np.array(df['PO3_cor']), np.array(df['Pair']), label=f"TRRM", color = cbl[k])
    # plt.plot(np.array(df['IM']), np.array(df['Pair']), label=f"Conventional", color = 'black')
    # # plt.plot(np.array(df['Ifast_deconv']), np.array(df['Pair']), label=f"TRRM (no iB0)", color = 'cyan')
    # plt.plot(np.array(df['I_OPM_jma']), np.array(df['Pair']), label=f"OPM", color = 'darkgreen')
    plt.plot(np.array(df['IM']), np.array(df['Pair']), label=f"IM")
    plt.plot(np.array(df['IM-iB0']), np.array(df['Pair']), label=f"IO3")
    plt.plot(np.array(df['Ifast_minib0_deconv']), np.array(df['Pair']), label=f"IFDS")



    # plt.plot(np.array(df['iB0']), np.array(df['Pair']), label=f"iB0")
    # plt.plot(np.array(df['Islow_conv']), np.array(df['Pair']), label=f"Islow_conv")




    ax0.legend(loc='best', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1)

    # ax0.axvline(x=0, color='grey', linestyle='--', linewidth=3)

    # 2nd grid TRRM
    ax1 = plt.subplot(gs[1])
    plt.yscale('log')
    plt.ylim([1000, 5])
    # plt.xlim([-3, 3])
    # # ax1.set_xticklabels(['',-2,-1,0,1,2,''])
    # plt.xticks([-2, -1, 0, 1, 2])

    plt.xlabel('Ratio', fontsize=size_label)
    plt.xticks(fontsize=size_label)

    ax1.tick_params(which='major', width=3)
    ax1.tick_params(which='minor', width=3)
    # ax1.xaxis.set_minor_locator(MultipleLocator(2))
    ax1.xaxis.set_tick_params(length=5, which='minor')
    ax1.xaxis.set_tick_params(length=10, which='major')
    ax1.yaxis.set_tick_params(length=5, which='minor')
    ax1.yaxis.set_tick_params(length=10, which='major')
    # plt.plot(df['ratio_2'], df['Pair'],  marker='o', label=f'2: iB0/IM', markersize=0.5)
    # plt.plot(df['ratio_3'], df['Pair'],   marker='o', label=f'3: IS/IM', markersize=0.5)
    # plt.plot(df['ratio_6'], df['Pair'],   marker='o', label=f'6: (IFDS - IF)/IM', markersize=0.5)
    # plt.plot(df['ratio_7'], df['Pair'],   marker='o', label=f'7: (P.E. kom - P.E. jma)/P.E. kom', markersize=0.5)
    # plt.plot(df['ratio_8'], df['Pair'],   marker='o', label=f'8: 2+3+6+7', markersize=0.5)

    plt.plot(df['ratio_2dc'], df['Pair'], marker='o', label=f'2: iB0/IFDS', markersize=0.5)
    plt.plot(df['ratio_3dc'], df['Pair'], marker='o', label=f'3: IS/IFDS', markersize=0.5)
    plt.plot(df['ratio_6ddc'], df['Pair'], marker='o', label=f'6d: IM/IFDS', markersize=0.5)

    plt.plot(df['ratio_6dc'], df['Pair'], marker='o', label=f'6: (IFDS - IF)/IFDS', markersize=0.5)
    # plt.plot(df['ratio_7'], df['Pair'], marker='o', label=f'7: (P.E. kom - P.E. jma)/P.E. kom', markersize=0.5)
    plt.plot(df['ratio_8dc'], df['Pair'], marker='o', label=f'8: 2+3+6d', markersize=0.5)


    ax1.set_yticklabels([])
    ax1.legend(loc='best', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1)

    # ax1.legend(loc='lower left', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1)
    # ax1.axvline(x=0, color='grey', linestyle='--', linewidth=3)
    plt.grid(True)

    # # # trrm + calibrated#
    ax2 = plt.subplot(gs[2])
    plt.yscale('log')
    plt.ylim([1000, 5])
    # plt.xlim([-3, 3])
    # # ax2.set_xticklabels(['',-2,-1,0,1,2,''])
    # plt.xticks([-2, -1, 0, 1, 2])
    #
    # plt.xlabel('Rel. Dif. [%]', fontsize=size_label)
    plt.xlabel('Ratio', fontsize=size_label)

    plt.xticks(fontsize=size_label)
    #
    ax2.tick_params(which='major', width=3)
    ax2.tick_params(which='minor', width=3)
    # ax2.xaxis.set_minor_locator(MultipleLocator(2))
    ax2.xaxis.set_tick_params(length=5, which='minor')
    ax2.xaxis.set_tick_params(length=10, which='major')
    ax2.yaxis.set_tick_params(length=5, which='minor')
    ax2.yaxis.set_tick_params(length=10, which='major')
    # plt.plot(df['ratio_2'], df['Pair'], marker='o', label=f'2: iB0/IM', markersize=0.5)
    # plt.plot(df['ratio_3'], df['Pair'], marker='o', label=f'3: IS/IM', markersize=0.5)
    # plt.plot(df['ratio_6b'], df['Pair'], marker='o', label=f'6b: IF/IM', markersize=0.5)
    # plt.plot(df['ratio_6d'], df['Pair'], marker='o', label=f'6d: IFDS/IM', markersize=0.5)
    # # plt.plot(df['ratio_7'], df['Pair'], marker='o', label=f'7: (P.E. kom - P.E. jma)/P.E. kom', markersize=0.5)
    # plt.plot(df['ratio_8b'], df['Pair'], marker='o', label=f'8b: 2+3+6b', markersize=0.5)
    # plt.plot(df['ratio_8d'], df['Pair'], marker='o', label=f'8d: 2+3+6d', markersize=0.5)

    plt.plot(df['ratio_2dc'], df['Pair'], marker='o', label=f'2: iB0/IFDS', markersize=0.5)
    plt.plot(df['ratio_3dc'], df['Pair'], marker='o', label=f'3: IS/IFDS', markersize=0.5)
    plt.plot(df['ratio_6bdc'], df['Pair'], marker='o', label=f'6b: IF/IFDS', markersize=0.5)
    plt.plot(df['ratio_6ddc'], df['Pair'], marker='o', label=f'6d: IM/IFDS', markersize=0.5)
    # plt.plot(df['ratio_7'], df['Pair'], marker='o', label=f'7: (P.E. kom - P.E. jma)/P.E. kom', markersize=0.5)
    plt.plot(df['ratio_8bdc'], df['Pair'], marker='o', label=f'8b: 2+3+6b', markersize=0.5)
    plt.plot(df['ratio_8ddc'], df['Pair'], marker='o', label=f'8d: 2+3+6d', markersize=0.5)
    ax2.set_yticklabels([])
    ax2.legend(loc='best', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1)

    # ax2.legend(loc='lower left', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1)
    # ax2.axvline(x=0, color='grey', linestyle='--', linewidth=3)
    plt.grid(True)
    #
    # ax3 = plt.subplot(gs[3])
    # plt.yscale('log')
    # plt.ylim([1000, 5])
    # # plt.xlim([-3, 3])
    # # # ax3.set_xticklabels(['',-2,-1,0,1,2,''])
    # # plt.xticks([-2, -1, 0, 1, 2])
    # #
    # # plt.xlabel(xreltitle, fontsize=size_label)
    # plt.xticks(fontsize=size_label)
    # plt.xlabel('Rel. Dif. [%]', fontsize=size_label)
    #
    # #
    # ax3.tick_params(which='major', width=3)
    # ax3.tick_params(which='minor', width=3)
    # ax3.xaxis.set_minor_locator(MultipleLocator(2))
    # ax3.xaxis.set_tick_params(length=5, which='minor')
    # ax3.xaxis.set_tick_params(length=10, which='major')
    # ax3.yaxis.set_tick_params(length=5, which='minor')
    # ax3.yaxis.set_tick_params(length=10, which='major')
    # # plt.plot(rdif_IM_deconv10[k], Yp, color=cbl[k], label=f'TRRM', linewidth=3)
    # # plt.plot(rdif_2b_IM_deconv10[k], Yp, color='skyblue', label=f'TRRM, 2*beta', linewidth=3)
    # #
    # # plt.plot(rdif_IM[k], Yp, color='black', label=f'Conventional', linewidth=3)
    # # plt.plot(df['RDif_ib0'], df['Pair'], marker='o', label=r'iB0 - IM', markersize=0.5, color='tab:blue')
    # # plt.plot(df['RDif_islow'], df['Pair'], marker='o', label=r'Islow - IM', markersize=0.5, color='tab:orange')
    # #
    # # plt.plot(df['RDif_ifdc'], df['Pair'], marker='o', label=r'I_{fast-iB0,D,S} - IM', markersize=0.5, color='tab:red')
    # # plt.plot(df['RDif_ifdc_noib0'], df['Pair'], marker='o', label=r'I_{fast,D,S} - IM', markersize=0.5,color='tab:green')
    # plt.plot(df['RDif_ib0_t'], df['Pair'], marker='o', label=r'iB0 - I_{fast-iB0,D,S}', markersize=0.5)
    # plt.plot(df['RDif_islow_t'], df['Pair'], marker='o', label=r'Islow - I_{fast-iB0,D,S}', markersize=0.5)
    # plt.plot(df['RDif_ifm_t'], df['Pair'], marker='o', label=r'Ifast - I_{fast-iB0,D,S}', markersize=0.5)
    #
    #
    # # 'RDif_ib0_t', 'RDif_islow_t', 'RDif_if_t', 'RDif_ifm_t'
    # ax3.set_yticklabels([])
    # ax3.legend(loc='best', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1)
    #
    # # ax3.legend(loc='lower left', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1)
    # ax3.axvline(x=0, color='grey', linestyle='--', linewidth=3)
    # plt.grid(True)

    plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/checks/plot2_1205_{year}_{labellist[k]}.png')

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