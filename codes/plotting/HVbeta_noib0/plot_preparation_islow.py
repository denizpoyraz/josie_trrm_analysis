import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from constant_variables import *


# df = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Deconv_preparationadded_all_updjma.csv")
# dfo = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Deconv_preparationadded_simulation_updjma.csv")
bool_minutes = True
tconv =60
# df = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Deconv_preparationadded_all_2023paper_sm_hv.csv")
# dfo = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Deconv_preparationadded_simulation_2023paper_sm_hv.csv")


df = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Deconv_preparationadded_all_2023paper_136_3_sm_hv.csv")
dfo = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Deconv_preparationadded_simulation_2023paper_136_3_sm_hv.csv")


df = df[(df.Sim == 136) & (df.Team == 3)]
dfo = dfo[(dfo.Sim == 136) & (dfo.Team == 3)]

simlist = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Sim'])
teamlist = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Team'])
sol = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Sol'].tolist())
buff = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Buf'])
ensci = np.asarray(df.drop_duplicates(['Sim', 'Team'])['ENSCI'])

dft = {}
dfto = {}

dft = {}
dfto = {}

# cbl = ['#e41a1c', '#a65628','#dede00', '#4daf4a', '#377eb8', '#984ea3']
# red, brown, yellow, green, blue, purple

size_label =22
size_title =24

print(simlist, teamlist)

for j in range(len(simlist)):

    if ensci[j] == 0:
        sondestr = 'SPC'
    else:
        sondestr = 'ENSCI'

    if sol[j] == 2.0: solstr = '2p0'
    if sol[j] == 1.0: solstr = '1p0'
    if sol[j] == 0.5: solstr = '0p5'

    if buff[j] == 0.1: bufstr = '0p1'
    if buff[j] == 0.5: bufstr = '0p5'
    if buff[j] == 1.0: bufstr = '1p0'

    if (ensci[j] == 1) & (sol[j] == 0.5) & (buff[j] == 0.5):
        beta = beta_en0505
        sst = 'SST 0.5'
    if (ensci[j] == 1) & (sol[j] == 1.0) & (buff[j] == 1.0):
        beta = beta_en1010
        sst = 'SST 1.0'
    if (ensci[j] == 0) & (sol[j] == 0.5) & (buff[j] == 0.5):
        beta = beta_sp0505
        sst = 'SST 0.5'
    if (ensci[j] == 0) & (sol[j] == 1.0) & (buff[j] == 1.0):
        beta = beta_sp1010
        sst = 'SST 1.0'


    title = str(simlist[j]) + '_' + str(teamlist[j])
            # + '_' + adxstr + sondestr + solstr + '-' + bufstr + 'B'
    type = sondestr + ' ' + str(sol[j]) + '\% - ' + str(buff[j]) + 'B'
    sp = str(simlist[j]) + '-' + str(teamlist[j])
    print(title)
    ptitle =  f'{sp} {sondestr} {sst} '


    ######
    df['Tsim_original'] = df['Tsim_original']/tconv
    df['Tsim'] = df['Tsim']/tconv
    dfo['Tsim'] = dfo['Tsim']/tconv

    dft[j] = df[(df.Sim == simlist[j]) & (df.Team == teamlist[j])]
    dfto[j] = dfo[(dfo.Sim == simlist[j]) & (dfo.Team == teamlist[j])]

    # # dft[j].TsimMin = dft[j].Tsim / 60
    # # dfto[j].TsimMin = dfto[j].Tsim / 60
    # dft[j].TsimMin = dft[j].Tsim /tconv
    # dfto[j].TsimMin = dfto[j].Tsim /tconv

    dft[j].time_ib1 = dft[j].time_ib1
    dfto[j].time_ib1 = dfto[j].time_ib1


    begin_sim =  df[df.Tsim_original >= 0]['Tsim'].tolist()[0]
    begin_sim=begin_sim
    dft[j]['Tsim'] = dft[j]['Tsim']
    print('begin_sim',begin_sim)
    print('time_ib1',dft[j].at[dft[j].first_valid_index(), 'time_ib1'])

    dft[j]['Tsim_new'] = dft[j]['Tsim'] - begin_sim
    dfto[j]['Tsim_new'] = dfto[j]['Tsim'] - begin_sim
    dft[j].time_ib1 = dft[j].time_ib1 - begin_sim
    dfto[j].time_ib1 = dfto[j].time_ib1  - begin_sim

    size_label = 22
    size_title = 24
    size_text = 18
    # ax1 = plt.subplot(figsize=(16,9))
    plt.figure(figsize=(12, 8))
    plt.ylabel(r'Current [$\mu$A])', fontsize=size_label)
    plt.xlabel(r'Simulation time [sec.]', fontsize=size_label)
    plt.yticks(fontsize=size_label)
    plt.xticks(fontsize=size_label)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())

    plt.gca().tick_params(which='major', width=2)
    plt.gca().tick_params(which='minor', width=2)
    plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
    plt.gca().xaxis.set_tick_params(length=5, which='minor')
    plt.gca().xaxis.set_tick_params(length=10, which='major')
    plt.gca().yaxis.set_tick_params(length=5, which='minor')
    plt.gca().yaxis.set_tick_params(length=10, which='major')



    if ((simlist[j] == 139) | (simlist[j] == 140) | (simlist[j] == 145) | (simlist[j] == 158) | (
            simlist[j] == 160) | (simlist[j] == 163) | (simlist[j] == 166)):
        plt.xlim(0, 250)
        if bool_minutes:
            plt.xlim(0, 45)

    plt.yscale('log')
    print(list(dft[j]))
    print(list(dfto[j]))

    t_ib1 = dft[j].at[dft[j].first_valid_index(), 'time_ib1']
    print('time ib1', t_ib1)
    #maybe better to plot everything in minutes
    if bool_minutes:
        # dft[j]['Tsim_new'] = dft[j]['Tsim_new']/tconv
        # t_ib1 = t_ib1/tconv
        plt.xlabel(r'Simulation time [min.]', fontsize=size_label)

    dftib1 = dft[j][dft[j].Tsim_new >= t_ib1]
    # dftib1 = dft[j]
    dft[j]['IM-iB0'] = dft[j]['IM'] - dft[j]['iB0']
    # plt.plot(dft[j].Tsim_new, dft[j].IM , label=r'I$_M$ - I$_{B0}$', color = cbl[4] ) #blue
    plt.plot(dft[j].Tsim_new, dft[j].IM - 0.03, label=r'I$_M$ - I$_{B0}$', color = cbl[4] ) #blue
    plt.plot(dft[j].Tsim_new, dft[j].I_slow_conv, label=r'I$_{S}$ (all range)',
             color = cbl[0], linestyle = '--', linewidth='2.0') #red
    plt.plot(dfto[j].Tsim_new, dfto[j].I_slow_convo, label=r'I$_{S}$ (simulation range)',
             color = cbl[1], linewidth='2.0')#brown
    # plt.plot(dfto[j].Tsim_new, dfto[j].I_slow_conv_ib1_decay, label=r'I$_{S}$ (real used)',
    #          color = 'cyan', linewidth='2.0')#brown


    # plt.plot(dftib1.Tsim_new, dftib1['I_slow_conv_ib1_decay_all'],
    #          label=r'I$_{S}$ (I$_{B1}$-I$_{B0})\times X_D$]', linewidth='2.0', color = cbl[3], linestyle = '--')#green
    plt.plot(dfto[j].Tsim_new, dfto[j].I_slow_conv_ib1_decay,
             label=r'I$_{S}$ (I$_{B1}$-I$_{B0})\times X_D$]', linewidth='2.0', color = cbl[3], linestyle = '--')#green
    plt.plot(dfto[j].Tsim_new, dfto[j].I_slow_conv_ib1_test,
             label=r'I$_{S}$ (I$_{B1}$-I$_{B0}$)', color = cbl[5], linewidth='1.5')#purple
    plt.plot(dft[j].Tsim_new, dft[j]['iB1'] - dft[j]['iB0'], label=r'I$_{B1}$-I$_{B0}$', color = cbl[2], linewidth='1.5')#yelloe
    # plt.axvline(x=dft[j].at[dft[j].first_valid_index(), 'time_ib1'], color='grey', linestyle=':', linewidth='2.0')

    # plt.xticks(np.arange(-4000, 7000, 1500))
    plt.ylim(0.0001, 8)
    plt.xlim(-4200, 7000)
    if bool_minutes:
        plt.xlim(-75, 120)
        plt.xticks(np.arange(-75, 150, 25))
    plt.title(ptitle, fontsize=size_title)
    plt.text((2165 - 200)/tconv, 1, 'RT1', weight='bold', fontsize=size_text, bbox={'facecolor': 'white', 'alpha': 0.6})
    plt.text((4100 - 200)/tconv, 4.5, 'RT2', weight='bold', fontsize=size_text, bbox={'facecolor': 'white', 'alpha': 0.6})
    plt.text((6200 - 175)/tconv, 4.5, 'RT3', weight='bold', fontsize=size_text, bbox={'facecolor': 'white', 'alpha': 0.6})
    # plt.text((8160 - 200)/tconv, 1.4, 'RT4', weight='bold', fontsize=size_text, bbox={'facecolor': 'white', 'alpha': 0.6})
    # plt.text((2155 - 200)/tconv, 1, 'RT1', weight='bold', fontsize=size_text)

    # plt.text(-58,0.0004, '1', weight='bold', fontsize=size_text, bbox={'facecolor': 'white', 'alpha': 0.6, 'boxstyle':'circle,pad=0.3'})
    plt.text(-62,0.0004, '1', weight='bold', fontsize=size_text, bbox={'facecolor': 'white', 'alpha': 0.4,'boxstyle':'circle,pad=0.3'})
    plt.text(-47,0.0004, '2', weight='bold', fontsize=size_text, bbox={'facecolor': 'white', 'alpha': 0.4,'boxstyle':'circle,pad=0.3'})
    plt.text(-0.75,0.5, '3', weight='bold', fontsize=size_text, bbox={'facecolor': 'white', 'alpha': 0.4,'boxstyle':'circle,pad=0.3'})
    plt.text(10,1.3, '4', weight='bold', fontsize=size_text, bbox={'facecolor': 'white', 'alpha': 0.4,'boxstyle':'circle,pad=0.3'})

    plt.legend(loc='lower right', frameon=True, fontsize='xx-large', markerscale=10, handletextpad=0.1 )

    plt.yscale('log')
    #
    plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v4/Fig_4_min_' + sp + '_RT3.png')
    # plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v3/Plots_PerSim/eps/Fig_4_min_' + sp + '.eps')
    # plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v3/Plots_PerSim/pdf/Fig_4__min' + sp + '.pdf')


    plt.show()
