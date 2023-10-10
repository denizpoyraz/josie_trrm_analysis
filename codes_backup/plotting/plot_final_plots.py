import numpy as np
import pandas as pd
# Libraries
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from data_cuts import cuts0910, cuts2017, cuts9602
from plotting_functions import filter_rdif, filter_rdif_all
from analyse_functions import Calc_average_Dif_yref,apply_calibration, cal_dif
from constant_variables import *
import warnings
warnings.filterwarnings("ignore")


#plot style variables#
# size_label = 28
# size_title = 32
# size_tick = 26
# size_legend = 18
size_label = 36
size_title = 40
size_tick = 34
size_legend = 32
bool_sm_vh = True
pre=''
if bool_sm_vh: pre = '_sm_hv'

bool_rdif = True
bool_two = True
bool_three = True
bool_triple = True
bool_adif = True
bool_test_2017 = False

df0910t = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_calibrated{pre}.csv", low_memory=False)

df0910 = df0910t.drop(['Unnamed: 0.1', 'index', 'Tact', 'Tair', 'Tinlet', 'TPint', 'I_Pump', 'VMRO3', 'VMRO3_OPM',
 'ADif_PO3S', 'RDif_PO3S','Z', 'Header_Team', 'Header_Sim', 'Header_PFunc', 'Header_PFcor',
 'Header_IB1', 'Simulator_RunNr', 'Date', 'Ini_Prep_Date', 'Prep_SOP', 'SerialNr', 'SerNr', 'Date_1',
 'SondeAge', 'Solutions', 'Volume_Cathode', 'ByPass_Cell', 'Current_10min_after_noO3', 'Resp_Time_4_1p5_sec',
'RespTime_1minOver2min', 'Final_BG', 'T100', 'mlOvermin', 'T100_post', 'mlOverminp1', 'RespTime_4_1p5_sec_p1','RespTime_1minOver2min_microamps', 'PostTestSolution_Lost_gr', 'PumpMotorCurrent', 'PumpMotorCurrent_Post',
 'PF_Unc', 'PF_Cor', 'BG', 'plog', 'Tcell', 'TcellC','Pw', 'massloss', 'Tboil', 'total_massloss', 'I_conv_slow',
                      'I_conv_slow_komhyr'], axis=1)

print(list(df0910))
df17 = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_calibrated{pre}.csv", low_memory=False)
df17 = df17[df17.iB2 >= 0]
df9602 = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie9602_calibrated{pre}.csv", low_memory=False)
df17['Year'] = '2017'
df0910['Year'] = '0910'
df9602['Year'] = '9602'

slist = [0,1,2,3,4,5]
y = [0] * 6

bool_current = False

# year = '2017'
# year_title = '2017'
# pyear = "2017"

year = '9602'
year_title = '1996/1998/2000/2002'
pyear = "9602"

# year = '0910'
# year_title = '2009/2010'
# pyear = "0910"

# year = 'all'
#
df = df0910
slist = [0,1,3,4]
prep='fig6'
pre1 = f'scatter_{prep}_'
pre2 = f'scatter_{pre}_'
bool_inter = False

if year == '2017':
    slist = [0,2,4,5]
    df = df17
    df = df[df.iB2 >= 0 ]
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
    slist = [0, 1, 2,3, 4, 5]
    pre1 = f'scatter_{pre}_'
    pre2 = f'fig13plus_{pre}_'


if not bool_current:

    ffile = '_pressure'


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
            dfc.loc[(dfc.Pair < 8) & (dfc.Pair > 4), clist[j]] = np.nan
        dfs = dfc[clist].interpolate()
        for v in vlist:
            dfs[v] = dfc[v]

        dfc = dfs.copy()
        for j in range(len(snlist)):
            dfc[snlist[j]] = 0
            dfc[snlist[j]] = dfs[clist[j]]

    df['ADif'], df['RDif'] = cal_dif(df, 'PO3_dqa', 'PO3_OPM', 'ADif', 'RDif')
    df['ADif_cor'], df['RDif_cor'] = cal_dif(df, 'PO3_cor', 'PO3_OPM', 'ADif_cor', 'RDif_cor')
    df['ADif_cal'], df['RDif_cal'] = cal_dif(df, 'PO3_cal', 'PO3_OPM', 'ADif_cal',
                                             'RDif_cal')
    profl = filter_rdif_all(df)
    # print(profl[1])


    adif_IM, adif_IM_err, rdif_IM, rdif_IM_err, Yp = Calc_average_Dif_yref(profl, 'PO3_dqa', 'PO3_OPM', 'pressure', yref)
    adif_IM_deconv10, adif_IM_deconv10_err, rdif_IM_deconv10, rdif_IM_deconv10_err, Yp = \
        Calc_average_Dif_yref(profl, 'PO3_cor', 'PO3_OPM', 'pressure', yref)
    adif_IM_cal, adif_IM_cal_err, rdif_IM_cal, rdif_IM_cal_err, Ypc = \
        Calc_average_Dif_yref(profl, 'PO3_cal', 'PO3_OPM', 'pressure', yref)

    if bool_inter:
        dfc['ADif'], dfc['RDif'] = cal_dif(dfc, 'PO3_dqa', 'PO3_OPM', 'ADif', 'RDif')
        dfc['ADif_cor'], dfc['RDif_cor'] = cal_dif(dfc, 'PO3_cor', 'PO3_OPM', 'ADif_cor', 'RDif_cor')
        dfc['ADif_cal'], dfc['RDif_cal'] = cal_dif(dfc, 'PO3_cal', 'PO3_OPM', 'ADif_cal',
                                                 'RDif_cal')
        proflc = filter_rdif_all(dfc)

        adif_IM, adif_IM_err, rdif_IM, rdif_IM_err, Yp = Calc_average_Dif_yref(proflc, 'PO3_dqa', 'PO3_OPM', 'pressure',
                                                                               yref)
        adif_IM_deconv10, adif_IM_deconv10_err, rdif_IM_deconv10, rdif_IM_deconv10_err, Yp = \
            Calc_average_Dif_yref(proflc, 'PO3_cor', 'PO3_OPM', 'pressure', yref)
        adif_IM_cal, adif_IM_cal_err, rdif_IM_cal, rdif_IM_cal_err, Ypc = \
            Calc_average_Dif_yref(proflc, 'PO3_cal', 'PO3_OPM', 'pressure', yref)

# profl[1].to_excel("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_EN1010_file.xlsx")
print('end')

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
    if bool_inter:
        proflc[k]['pair_nan'] = 0
        proflc[k].loc[proflc[k].Pair.isnull(), 'pair_nan'] = 1

        dft = proflc[k][proflc[k].pair_nan == 0]

    dft = dft[(dft.RDif_cor < urdif) & (dft.RDif_cor > lrdif)]
    
    filt_p = dft.Pair >= pcut
    y[k] = np.array(dft[filt_p]['Pair'])
    labelc[k] = b[k]
    if pyear == '0910':
        a_year[k] = a_0910[k]
        b_year[k] = b_0910[k]
        labelc[k] = b_0910[k]
        if bool_sm_vh:
            a_year[k] = a_0910_smhv[k]
            b_year[k] = b_0910_smhv[k]
            labelc[k] = b_0910_smhv[k]


    if pyear == '2017':
        a_year[k] = a_2017[k]
        b_year[k] = b_2017[k]
        labelc[k] = b_2017[k]
        if bool_sm_vh:
            a_year[k] = a_2017_smhv[k]
            b_year[k] = b_2017_smhv[k]
            labelc[k] = b_2017_smhv[k]

    if pyear == '9602':
        a_year[k] = a_9602[k]
        b_year[k] = b_9602[k]
        labelc[k] = b_9602[k]
        if bool_sm_vh:
            a_year[k] = a_9602_smhv[k]
            b_year[k] = b_9602_smhv[k]
            labelc[k] = b_9602_smhv[k]


    if  pyear == "0910-2017":
        a_year[k] = a[k]
        b_year[k] = b[k]
        labelc[k] = b[k]
        if bool_sm_vh:
            a_year[k] = a_smhv[k]
            b_year[k] = b_smhv[k]
            labelc[k] = b_smhv[k]



    if b_year[k] < 0:
        sign[k] = '-'
        labelc[k] = -1 * b_year[k]
    # if b[k] < 0:
    #     sign[k] = '-'
    #     labelc[k] = -1 * b[k]

    plotname = f'{pre1}{pyear}_{labellist_n[k]}'

    nsim[k] = len(profl[k].drop_duplicates(['Sim', 'Team']))

    maintitle = 'JOSIE ' + year_title + " " + labellist[k]
    ytitle = 'Pressure [hPa]'
    xtitle = 'Current'
    # xrtitle = 'Rel. Differences [%] \n (Sonde - OPM)/OPM'
    xrtitle = '(Sonde - OPM)/OPM [%]'

    if bool_rdif:
        if bool_two:
            fig = plt.figure(figsize=(17, 16))
            # plt.suptitle("GridSpec Inside GridSpec")
            plt.suptitle(maintitle, fontsize=size_title, y = 0.93)

            gs = gridspec.GridSpec(1, 2, width_ratios=[2, 2])
            gs.update(wspace=0.0005, hspace=0.05)
            ax0 = plt.subplot(gs[0])
            for axis in ['top', 'bottom', 'left', 'right']:
                ax0.spines[axis].set_linewidth(1.5)  # change width
            plt.yscale('log')
            plt.ylim([1000, 5])
            plt.xlim([-40, 40])


            plt.grid(True)
            #
            ax0.set_xticklabels(['', '', -20, '', 0, '', 20, '', ''])
            ax0.yaxis.set_major_formatter(ScalarFormatter())
            plt.xlabel(xrtitle, fontsize=size_label)
            plt.ylabel(ytitle, fontsize=size_label, labelpad=-15)
            plt.xticks(fontsize=size_tick)
            plt.yticks(fontsize=size_tick)
            plt.grid(True)
            plt.gca().tick_params(which='major', width=5)
            plt.gca().tick_params(which='minor', width=5)
            # plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
            plt.gca().xaxis.set_tick_params(length=5, which='minor')
            plt.gca().yaxis.set_tick_params(length=5, which='minor')
            plt.gca().xaxis.set_tick_params(length=10, which='major')
            plt.gca().yaxis.set_tick_params(length=10, which='major')


            plt.plot(profl[k]['RDif'], profl[k]['Pair'], color=cbl[k], linestyle='None', marker='o', markersize=0.5,
                     label = 'Conventional')
            ax0.errorbar(rdif_IM[k], Yp, xerr=rdif_IM_err[k], color='black', linewidth=3, elinewidth=1, capsize=1, capthick=1)
            ax0.plot([], [], ' ', label=r"N$_{sim}$"+f"={nsim[k]}")


            ax0.legend(loc='upper left', frameon=True, fontsize=size_legend, markerscale=25,  handletextpad=0.01, framealpha=0.65)

            ax0.axvline(x=0, color='grey', linestyle='--', linewidth=3)

            # 2nd grid TRC
            ax1 = plt.subplot(gs[1])
            for axis in ['top', 'bottom', 'left', 'right']:
                ax1.spines[axis].set_linewidth(1.5)  # change width
            plt.yscale('log')
            plt.ylim([1000, 5])
            plt.xlim([-40, 40])

            ax1.set_xticklabels(['', '', -20, '', 0, '', 20, '', ''])
            ax1.yaxis.set_major_formatter(ScalarFormatter())
            plt.xlabel(xrtitle, fontsize=size_label)
            # plt.ylabel(ytitle, fontsize=size_label, labelpad=-15)

            plt.grid(True)
            ax1.set_xticklabels(['', '', -20, '', 0, '', 20, '', ''])
            # ax0.yaxis.set_major_formatter(ScalarFormatter())
            plt.xlabel(xrtitle, fontsize=size_label)
            # plt.ylabel(ytitle, fontsize=size_label, labelpad=-15)
            plt.xticks(fontsize=size_tick)
            # plt.yticks(fontsize=size_tick)
            plt.grid(True)
            plt.gca().tick_params(which='major', width=5)
            plt.gca().tick_params(which='minor', width=5)
            # plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
            plt.gca().xaxis.set_tick_params(length=5, which='minor')
            plt.gca().yaxis.set_tick_params(length=5, which='minor')
            plt.gca().xaxis.set_tick_params(length=10, which='major')
            plt.gca().yaxis.set_tick_params(length=10, which='major')


            plt.plot(profl[k]['RDif_cor'], profl[k]['Pair'], color=cbl[k], linestyle='None', marker='o',
                     label=f'TRC', markersize=0.5)


            plt.plot(a_year[k] + b_year[k] * np.log10(y[k]), y[k], color='black', linestyle=':', linewidth=6,
                     label=rf'{round(a_year[k], 2)} {sign[k]} {round(labelc[k], 2)}$\cdot$log(P)')
            ax1.errorbar(rdif_IM_deconv10[k], Yp, xerr=rdif_IM_deconv10_err[k],
                         color='black', linewidth=3, elinewidth=1, capsize=1, capthick=1)

            ax1.set_yticklabels([])

            ax1.legend(loc='upper left', frameon=True, fontsize=size_legend, markerscale=25, handletextpad=0.05,framealpha=0.65)

            ax1.axvline(x=0, color='grey', linestyle='--', linewidth=3)
            plt.grid(True)

            plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v7/png/{plotname}{ffile}.png')
            plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v7/eps/{plotname}{ffile}.eps')
            plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v7/pdf/{plotname}{ffile}.pdf')
            # plt.show()
            plt.close()

        if bool_three:
            plotname = f'v2_RDif_{pre1}{pyear}_{labellist_n[k]}_Scatter'

            nsim[k] = len(profl[k].drop_duplicates(['Sim', 'Team']))

            maintitle = 'JOSIE ' + year_title + " " + labellist[k]
            ytitle = 'Pressure [hPa]'
            # xtitle = 'Current'
            xrtitle = '(Sonde - OPM)/OPM [%]'

            fig = plt.figure(figsize=(24, 14))
            # plt.suptitle("GridSpec Inside GridSpec")
            plt.suptitle(maintitle, fontsize=size_title, y=0.93)

            # gs = gridspec.GridSpec(1, 2, width_ratios=[2, 2])
            gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 2])

            gs.update(wspace=0.0005, hspace=0.05)
            ax0 = plt.subplot(gs[0])
            for axis in ['top', 'bottom', 'left', 'right']:
                ax0.spines[axis].set_linewidth(1.5)  # change width
            plt.yscale('log')
            plt.ylim([1000, 5])
            plt.xlim([-40, 40])

            ax0.set_xticklabels(['', '', -20, '', 0, '', 20, '', ''])
            # ax0.yaxis.set_major_formatter(ScalarFormatter())

            plt.xlabel(xrtitle, fontsize=size_label)
            plt.ylabel(ytitle, fontsize=size_label, labelpad=-5)
            plt.xticks(fontsize=size_tick)
            plt.yticks(fontsize=size_tick)
            plt.grid(True)
            plt.gca().tick_params(which='major', width=5)
            plt.gca().tick_params(which='minor', width=5)
            # plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
            plt.gca().xaxis.set_tick_params(length=5, which='minor')
            plt.gca().yaxis.set_tick_params(length=5, which='minor')
            plt.gca().xaxis.set_tick_params(length=10, which='major')
            plt.gca().yaxis.set_tick_params(length=10, which='major')

            plt.plot(profl[k]['RDif'], profl[k]['Pair'], color=cbl[k], linestyle='None', marker='o', markersize=0.5,
                     label='Conventional')
            ax0.errorbar(rdif_IM[k], Yp, xerr=rdif_IM_err[k], color='black', linewidth=3, elinewidth=1, capsize=1,
                         capthick=1)
            ax0.plot([], [], ' ', label=r"N$_{sim}$="+f"{nsim[k]}")

            ax0.legend(loc='upper left', frameon=True, fontsize=size_legend, markerscale=25, handletextpad=0.1,framealpha=0.65)

            ax0.axvline(x=0, color='grey', linestyle='--', linewidth=3)

            # 2nd grid TRC
            ax1 = plt.subplot(gs[1])
            for axis in ['top', 'bottom', 'left', 'right']:
                ax1.spines[axis].set_linewidth(1.5)  # change width
            plt.yscale('log')
            plt.ylim([1000, 5])
            plt.xlim([-40, 40])
            ax1.set_xticklabels(['', '', -20, '', 0, '', 20, '', ''])
            ax1.yaxis.set_major_formatter(ScalarFormatter())

            plt.xlabel(xrtitle, fontsize=size_label)
            # plt.ylabel(ytitle, fontsize=size_label, labelpad=-15)
            plt.xticks(fontsize=size_tick)
            # plt.yticks(fontsize=size_tick)
            plt.grid(True)
            plt.gca().tick_params(which='major', width=5)
            plt.gca().tick_params(which='minor', width=5)
            # plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
            plt.gca().xaxis.set_tick_params(length=5, which='minor')
            plt.gca().yaxis.set_tick_params(length=5, which='minor')
            plt.gca().xaxis.set_tick_params(length=10, which='major')
            plt.gca().yaxis.set_tick_params(length=10, which='major')
            plt.plot(profl[k]['RDif_cor'], profl[k]['Pair'], color=cbl[k], linestyle='None', marker='o',
                     label=f'TRC', markersize=0.5)
            ax1.errorbar(rdif_IM_deconv10[k], Yp, xerr=rdif_IM_deconv10_err[k],
                         color='black', linewidth=3, elinewidth=1, capsize=1, capthick=1)

            ax1.set_yticklabels([])
            ax1.legend(loc='upper left', frameon=True, fontsize=size_legend, markerscale=25, handletextpad=0.1,framealpha=0.65)

            # ax1.legend(loc='lower left', frameon=True, fontsize=size_label, markerscale=25, handletextpad=0.1)
            ax1.axvline(x=0, color='grey', linestyle='--', linewidth=3)
            plt.grid(True)

            # trrm + calibrated#
            ax2 = plt.subplot(gs[2])
            for axis in ['top', 'bottom', 'left', 'right']:
                ax2.spines[axis].set_linewidth(1.5)  # change width
            plt.yscale('log')
            plt.ylim([1000, 5])
            plt.xlim([-40, 40])

            ax2.set_xticklabels(['', '', -20, '', 0, '', 20, '', ''])
            ax2.yaxis.set_major_formatter(ScalarFormatter())

            plt.xlabel(xrtitle, fontsize=size_label)
            # plt.ylabel(ytitle, fontsize=size_label, labelpad=-15)
            plt.xticks(fontsize=size_tick)
            # plt.yticks(fontsize=size_tick)
            plt.grid(True)
            plt.gca().tick_params(which='major', width=5)
            plt.gca().tick_params(which='minor', width=5)
            # plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
            plt.gca().xaxis.set_tick_params(length=5, which='minor')
            plt.gca().yaxis.set_tick_params(length=5, which='minor')
            plt.gca().xaxis.set_tick_params(length=10, which='major')
            plt.gca().yaxis.set_tick_params(length=10, which='major')




            plt.plot(profl[k]['RDif_cal'], profl[k]['Pair'], color=cbl[k], linestyle='None', marker='o',
                     label=f'TRCC', markersize=0.5)
            ax2.errorbar(rdif_IM_cal[k], Yp, xerr=rdif_IM_cal_err[k],
                         color='black', linewidth=3, elinewidth=1, capsize=1, capthick=1)

            ax2.set_yticklabels([])
            ax2.legend(loc='upper left', frameon=True, fontsize=size_legend, markerscale=25, handletextpad=0.1,framealpha=0.65)

            # ax2.legend(loc='lower left', frameon=True, fontsize=size_label, markerscale=25, handletextpad=0.1)
            ax2.axvline(x=0, color='grey', linestyle='--', linewidth=3)
            plt.grid(True)

            plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v7/png/{plotname}{ffile}.png')
            plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v7/eps/{plotname}{ffile}.eps')
            plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v7/pdf/{plotname}{ffile}.pdf')
            # plt.show()

            plt.close()

    #now plot also in ADif plots
    #now plot also in ADif plots
    #now plot also in ADif plots
    #now plot also in ADif plots
    if bool_adif:

        plotname = f'v2_ADif_{pre1}{pyear}_{labellist_n[k]}_Scatter'

        nsim[k] = len(profl[k].drop_duplicates(['Sim', 'Team']))

        maintitle = 'JOSIE ' + year_title + " " + labellist[k]
        ytitle = 'Pressure [hPa]'
        # xtitle = 'Current'
        xrtitle = 'Sonde - OPM [mPa]'

        fig = plt.figure(figsize=(22, 14))
        # plt.suptitle("GridSpec Inside GridSpec")
        plt.suptitle(maintitle, fontsize=size_title, y = 0.93)

        # gs = gridspec.GridSpec(1, 2, width_ratios=[2, 2])
        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 2])

        gs.update(wspace=0.0005, hspace=0.05)
        ax0 = plt.subplot(gs[0])
        for axis in ['top', 'bottom', 'left', 'right']:
            ax0.spines[axis].set_linewidth(1.5)  # change width
        plt.yscale('log')
        plt.ylim([1000, 5])
        plt.xlim([-3, 3])
        plt.xticks([-2, -1, 0, 1, 2])

        plt.yticks(fontsize=size_label)
        plt.xticks(fontsize=size_label)

        ax0.yaxis.set_major_formatter(ScalarFormatter())

        plt.grid(True)
        plt.xlabel(xrtitle, fontsize=size_label)
        plt.ylabel(ytitle, fontsize=size_label)
        ax0.tick_params(which='major', width=3)
        ax0.tick_params(which='minor',width=3)
        # ax0.xaxis.set_minor_locator(MultipleLocator(2))
        # ax0.xaxis.set_tick_params(length=5, which='minor')
        ax0.xaxis.set_tick_params(length=10, which='major')
        ax0.yaxis.set_tick_params(length=5, which='minor')
        ax0.yaxis.set_tick_params(length=10, which='major')

        plt.plot(profl[k]['ADif'], profl[k]['Pair'], color=cbl[k], linestyle='None', marker='o', markersize=0.5,
                 label = 'Conventional')
        ax0.errorbar(adif_IM[k], Yp, xerr=adif_IM_err[k], color='black', linewidth=3, elinewidth=1, capsize=1, capthick=1)
        ax0.plot([], [], ' ', label=r"N$_{sim}$="+f"{nsim[k]}")


        ax0.legend(loc='upper left', frameon=True, fontsize=size_legend, markerscale=25,  handletextpad=0.1,framealpha=0.65)

        ax0.axvline(x=0, color='grey', linestyle='--', linewidth=3)

        # 2nd grid TRC
        ax1 = plt.subplot(gs[1])
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(1.5)  # change width
        plt.yscale('log')
        plt.ylim([1000, 5])
        plt.xlim([-3, 3])
        # ax1.set_xticklabels(['',-2,-1,0,1,2,''])
        plt.xticks([-2, -1, 0, 1, 2])

        plt.xlabel(xrtitle, fontsize=size_label)
        plt.xticks(fontsize=size_label)

        ax1.tick_params(which='major', width=3)
        ax1.tick_params(which='minor', width=3)
        ax1.xaxis.set_minor_locator(MultipleLocator(2))
        ax1.xaxis.set_tick_params(length=5, which='minor')
        ax1.xaxis.set_tick_params(length=10, which='major')
        ax1.yaxis.set_tick_params(length=5, which='minor')
        ax1.yaxis.set_tick_params(length=10, which='major')
        plt.plot(profl[k]['ADif_cor'], profl[k]['Pair'], color=cbl[k], linestyle='None', marker='o',
                 label=f'TRC', markersize=0.5)
        ax1.errorbar(adif_IM_deconv10[k], Yp, xerr=adif_IM_deconv10_err[k],
                     color='black', linewidth=3, elinewidth=1, capsize=1, capthick=1)

        ax1.set_yticklabels([])
        ax1.legend(loc='upper left', frameon=True, fontsize=size_legend, markerscale=25, handletextpad=0.1,framealpha=0.65 )

        # ax1.legend(loc='lower left', frameon=True, fontsize=size_label, markerscale=25, handletextpad=0.1)
        ax1.axvline(x=0, color='grey', linestyle='--', linewidth=3)
        plt.grid(True)

        # trrm + calibrated#
        ax2 = plt.subplot(gs[2])
        for axis in ['top', 'bottom', 'left', 'right']:
            ax2.spines[axis].set_linewidth(1.5)  # change width
        plt.yscale('log')
        plt.ylim([1000, 5])
        plt.xlim([-3, 3])
        # ax2.set_xticklabels(['',-2,-1,0,1,2,''])
        plt.xticks([-2, -1, 0, 1, 2])

        plt.xlabel(xrtitle, fontsize=size_label)
        plt.xticks(fontsize=size_label)

        ax2.tick_params(which='major', width=3)
        ax2.tick_params(which='minor', width=3)
        ax2.xaxis.set_minor_locator(MultipleLocator(2))
        ax2.xaxis.set_tick_params(length=5, which='minor')
        ax2.xaxis.set_tick_params(length=10, which='major')
        ax2.yaxis.set_tick_params(length=5, which='minor')
        ax2.yaxis.set_tick_params(length=10, which='major')
        plt.plot(profl[k]['ADif_cal'], profl[k]['Pair'], color=cbl[k], linestyle='None', marker='o',
                 label=f'TRCC', markersize=0.5)
        ax2.errorbar(adif_IM_cal[k], Yp, xerr=adif_IM_cal_err[k],
                     color='black', linewidth=3, elinewidth=1, capsize=1, capthick=1)

        ax2.set_yticklabels([])
        ax2.legend(loc='upper left', frameon=True, fontsize=size_legend, markerscale=25, handletextpad=0.1,framealpha=0.65)

        # ax2.legend(loc='lower left', frameon=True, fontsize=size_label, markerscale=25, handletextpad=0.1)
        ax2.axvline(x=0, color='grey', linestyle='--', linewidth=3)
        plt.grid(True)
        
        plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v7/png/{plotname}{ffile}.png')
        plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v7/eps/{plotname}{ffile}.eps')
        plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v7/pdf/{plotname}{ffile}.pdf')
        # plt.show()

        plt.close()



# PLOT ALL in ONE

if bool_triple:

    for k in slist:

        a[k] = a[k]
        plotname = f'{pre2}{pyear}_{labellist_n[k]}_Triple'


        nsim[k] = len(profl[k].drop_duplicates(['Sim', 'Team']))

        maintitle = 'JOSIE ' + year_title + " " + labellist[k]
        ytitle = 'Pressure [hPa]'
        xtitle = 'Current'
        xrtitle = 'Rel. Differences [%] \n (Sonde - OPM)/OPM'

        # fig, ax = plt.subplots(figsize=(8, 12), layout = 'constrained')
        fig, ax = plt.subplots(figsize=(9, 12))
        plt.suptitle(maintitle, fontsize=size_title, y = 0.93)
        plt.ylabel(ytitle, fontsize=size_label  )

        plt.yscale('log')
        plt.ylim([1000, 5])
        # plt.xlim([-15,15])
        plt.xlim([-40, 40])
        plt.yticks(fontsize=size_label)
        plt.xticks(fontsize=size_label)
        ax.tick_params(which='major', width=3)
        ax.tick_params(which='minor', width=3)
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.xaxis.set_tick_params(length=5, which='minor')
        ax.xaxis.set_tick_params(length=10, which='major')
        ax.yaxis.set_tick_params(length=5, which='minor')
        ax.yaxis.set_tick_params(length=10, which='major')

        ax.yaxis.set_label_coords(x = - 0.115, y=0.5)

        ax.yaxis.set_major_formatter(ScalarFormatter())
        plt.grid(True)
        plt.xlabel(xrtitle, fontsize=size_label)

        ax.errorbar(rdif_IM[k], Yp, xerr=rdif_IM_err[k], label=f'Conventional',
                     color=cbl[0], linewidth=2, elinewidth=1, capsize=1, capthick=1)

        ax.errorbar(rdif_IM_deconv10[k], Yp, xerr=rdif_IM_deconv10_err[k], label=f'TRC',
                     color=cbl[4], linewidth=2, elinewidth=1, capsize=1, capthick=1)

        ax.errorbar(rdif_IM_cal[k], Yp, xerr=rdif_IM_cal_err[k], label=f'TRCC',
                     color=cbl[3], linewidth=2, elinewidth=1, capsize=1, capthick=1)

        ax.legend(loc='upper left', frameon=True, fontsize=size_legend, markerscale=3)

        ax.axvline(x=0, color='grey', linestyle='--', linewidth=3)

        # plt.tight_layout()

        plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v7/png/{plotname}{ffile}.png')
        plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v7/eps/{plotname}{ffile}.eps')
        plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v7/pdf/{plotname}{ffile}.pdf')
        # plt.show()

        plt.close()
##################################################################################################################


 # handles, labels = ax0.get_legend_handles_labels()
    # order = [0, 2, 1]
    # ax0.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
    #            loc='upper left', frameon=True, fontsize='x-large', markerscale=7 )
    # ax0.legend(loc='lower left', frameon=True, fontsize=size_label, markerscale=25, handletextpad=0.05)

if bool_current:

    ffile = '_current'

    # df['I_OPM_jma'] = (df['PO3_OPM'] * df['PFcor_jma']) / (df['TPext'] * 0.043085)

    df['IminusiB1'] = df['IM'] - df['iB1']
    if year == '2017': df['IminusiB1'] = df['IM'] - df['iB2']
    if year == 'all':
        df.loc[df.Year == '2017', 'IminusiB1'] = df.loc[df.Year == '2017', 'IM'] - df.loc[df.Year == '2017', 'iB2']
        df.loc[df.Year == '0910', 'IminusiB1'] = df.loc[df.Year == '0910', 'IM'] - df.loc[df.Year == '0910', 'iB1']

    # df['ADif'], df['RDif'] = cal_dif(df, 'IM', 'I_OPM_kom', 'ADif', 'RDif')
    df['ADif'], df['RDif'] = cal_dif(df, 'IminusiB1', 'I_OPM_kom', 'ADif', 'RDif')
    df['ADif_cor'], df['RDif_cor'] = cal_dif(df, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma', 'ADif_cor', 'RDif_cor')
    df['ADif_cal'], df['RDif_cal'] = cal_dif(df, 'I_corrected', 'I_OPM_jma', 'ADif_cal',
                                             'RDif_cal')

    profl = filter_rdif_all(df)

    # adif_IM, adif_IM_err, rdif_IM, rdif_IM_err, Yp = Calc_average_Dif_yref(profl, 'IM', 'I_OPM_kom','pressure', yref)
    adif_IM, adif_IM_err, rdif_IM, rdif_IM_err, Yp = Calc_average_Dif_yref(profl, 'IminusiB1', 'I_OPM_kom',
                                                                           'pressure', yref)

    adif_IM_deconv10, adif_IM_deconv10_err, rdif_IM_deconv10, rdif_IM_deconv10_err, Yp = \
        Calc_average_Dif_yref(profl, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma', 'pressure', yref)
    adif_IM_cal, adif_IM_cal_err, rdif_IM_cal, rdif_IM_cal_err, Ypc = \
        Calc_average_Dif_yref(profl, 'I_corrected', 'I_OPM_jma', 'pressure', yref)
