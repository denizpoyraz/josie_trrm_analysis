import numpy as np
import pandas as pd
from scipy import stats
# Libraries
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from data_cuts import cuts0910, cuts2017, cuts9602
from homogenization_functions import pumpflow_efficiency, return_phipcor, VecInterpolate_log
from constant_variables import *
from analyse_functions import Calc_average_profile_pressure
from plotting_functions import filter_rdif, filter_rdif_all


df17c = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_deconv_2023paper.csv", low_memory=False)

df09c = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/"
                    "Josie0910_deconv_2023_unitedpaper.csv", low_memory=False)

# print(list(df09c))

# df09c['Ifast_minib0_deconv_ib1_decay_sm10'] = df09c['Ifast_minib0_deconv_ib1_decay'].rolling(window=5,center=True).mean()
df09c['Ifast_minib0_deconv_sm10'] = df09c['Ifast_minib0_deconv_ib1_decay'].rolling(window=5, center=True).mean()
df17c['Ifast_minib0_deconv_sm10'] = df17c['Ifast_minib0_deconv'].rolling(window=5, center=True).mean()

df09 = cuts0910(df09c)
df17 = cuts2017(df17c)
# df96 = cuts9602(df96c)

df09['Year'] = '0910'
df17['Year'] = '2017'
# df96['Year'] = '9602'

dfa = pd.concat([df09, df17], ignore_index=True)
# year = '2009,2010,2017'
year = '0910'
# year = '2017'
df = dfa[dfa.Year == year]
# 
# prof = filter_rdif_all(dfo)
# # prof = [profEN0505, profEN1010, profEN1001, profSP0505, profSP1010, profSP1001]
# beta_l = [beta_en0505,beta_en1010,beta_1001, beta_sp0505,beta_sp1010, beta_1001]
# i = 0
# beta = beta_l[i]
# 
# df = prof[i]
# 
# # df_av =
# ['IM', ]
# Calc_average_profile_pressure(df, 'IM')
# print(list(df))
#########################################################

simlist = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Sim'])
teamlist = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Team'])
sol = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Sol'].tolist())
buff = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Buf'])
ensci = np.asarray(df.drop_duplicates(['Sim', 'Team'])['ENSCI'])

# dft = df[(df.Sim == 136) & (df.Team ==3)]
dft = {}

for j in range(len(simlist)):

    sondestr = ''
    adxstr = ''
    solstr = ''
    bufstr = ''

    af = 1
    betac = 0

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
        betac = beta_en0505;
        ac = a[0];
        bc = b[0];
        ac_err = a_err[0];
        bc_err = b_err[0]
    if (ensci[j] == 1) & (sol[j] == 1.0) & (buff[j] == 1.0):
        betac = beta_en1010;
        ac = a[1];
        bc = b[1];
        ac_err = a_err[1];
        bc_err = b_err[1]
    if (ensci[j] == 0) & (sol[j] == 0.5) & (buff[j] == 0.5):
        betac = beta_sp0505;
        ac = a[3];
        bc = b[3];
        ac_err = a_err[3];
        bc_err = b_err[3]
    if (ensci[j] == 0) & (sol[j] == 1.0) & (buff[j] == 1.0):
        betac = beta_sp1010;
        ac = a[4];
        bc = b[4];
        ac_err = a_err[4];
        bc_err = b_err[4]
    if (ensci[j] == 1) & (sol[j] == 1.0) & (buff[j] == 0.1):
        betac = 0.023;
        ac = a[2];
        bc = b[2];
        ac_err = a_err[2];
        bc_err = b_err[2]
    if (ensci[j] == 0) & (sol[j] == 1.0) & (buff[j] == 0.1):
        betac = 0.023;
        ac = a[5];
        bc = b[5];
        ac_err = a_err[5];
        bc_err = b_err[5]

    title = str(simlist[j]) + '_' + str(teamlist[j]) + '_' + adxstr + sondestr + solstr + '-' + bufstr + 'B'
    type = sondestr + ' ' + str(sol[j]) + '\% - ' + str(buff[j]) + 'B'
    sp = str(simlist[j]) + '-' + str(teamlist[j])
    ptitle = sp + ' ' + sondestr + ' ' + str(sol[j]) + '% - ' + str(buff[j]) + 'B'
    print(title)

    dft[j] = df[(df.Sim == simlist[j]) & (df.Team == teamlist[j])]
    ### for data of every 12 seconds
    dft[j]['timef'] = pd.to_datetime(dft[j]["Tsim"], unit='s')

    dft[j] = dft[j].resample('5T', on='timef').mean().interpolate()
    # df = df.reset_index()
    dft[j] = dft[j].reset_index()
    dft[j]['dI'] = 0
    dft[j].loc[dft[j].IM < 1, 'dI'] = 0.005
    dft[j].loc[dft[j].IM >= 1, 'dI'] = 0.5 / 100 * dft[j].loc[dft[j].IM > 1, 'IM']
    dft[j]['dib1'] = 0.02
    dft[j]['cPL'] = 0.007
    dft[j]['dcPL'] = 0.002
    unc_cPL = 0.002
    dft[j]['cPH'] = 0.02
    dft[j]['dcPH'] = 0.002
    unc_cPH = 0.002
    if year == '2017':
        dft[j]['cPH'] = 0.03
        dft[j]['dcPH'] = 0.003
        unc_cPH = 0.003
    
    opm_err = 0.02
    
    dft[j]['eta_c'] = 1
    dft[j]['deta_c'] = 0.03
    dft[j]['eta_a'] = 1
    dft[j]['deta_a'] = 0.01
    dft[j]['dtpump'] = 0.7
    
    dft[j]['dbeta'] = 0.005
    
    
    dft[j]['dPhim'] = 0.01
    dft[j]['Phip_ground'] = dft[j]['PFcor']
    dft[j]['unc_Phip_ground'] = dft[j]['Phip_ground'] * np.sqrt(
        (dft[j]['dPhim']) ** 2 + (unc_cPL) ** 2 + (unc_cPH) ** 2)
    
    dft[j]['Phip_cor_kom'], dft[j]['unc_Phip_cor_kom'] = return_phipcor(dft[j], 'Phip_ground', 'unc_Phip_ground',
                                                                        'Cpf_kom', 'unc_Cpf_kom')
    dft[j]['Phip_cor_jma'], dft[j]['unc_Phip_cor_jma'] = return_phipcor(dft[j], 'Phip_ground', 'unc_Phip_ground',
                                                                        'Cpf_jma', 'unc_Cpf_jma')
    
    print('two')
    
    dft[j]['PO3_dqa'] = 0.043085 * dft[j]['Tpump_cor'] * (dft[j]['IM'] - dft[j]['iB0']) / \
                        (1 * dft[j]['Phip_cor_kom'])
    
    dft[j]['PO3_trrm'] = 0.043085 * dft[j]['Tpump_cor'] * (dft[j]['Ifast_minib0_deconv_sm10']) / \
                         (1 * dft[j]['Phip_cor_jma'])
    
    # a
    dft[j]['d_im_bkg'] = (dft[j]['dI'] ** 2 + dft[j]['dib1'] ** 2) / ((dft[j]['IM'] - dft[j]['iB0']) ** 2)
    dft[j]['d_im_bkg_trm'] = (dft[j]['dI'] ** 2 + dft[j]['dib1'] ** 2 + dft[j]['dbeta'] ** 2) / \
                             ((dft[j]['IM'] - dft[j]['iB0'] - betac) ** 2)
    
    # b
    dft[j]['d_pfe_hum'] = ((dft[j]['unc_Cpf_kom'] / dft[j]['Cpf_kom']) ** 2) + \
                          ((dft[j]['dPhim'] * dft[j]['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2))
    # c

    dft[j]['d_eta_c'] = (dft[j]['deta_c'] / dft[j]['eta_c']) ** 2
    dft[j]['d_eta_c_trm'] = ((ac_err**2 + (np.log10(dft[j]['Pair']) * bc_err)**2)/(1 + ac + bc * np.log10(dft[j]['Pair']))**2) + opm_err**2
    # d
    dft[j]['d_eta_a'] = (dft[j]['deta_a'] / dft[j]['eta_a']) ** 2
    # e
    dft[j]['d_tpump'] = (dft[j]['dtpump'] / dft[j]['Tpump_cor']) ** 2
    
    dft[j]['tota_unc'] = \
        np.sqrt(dft[j]['d_im_bkg'] + dft[j]['d_pfe_hum'] + dft[j]['d_eta_c'] + dft[j]['d_eta_a'] + dft[j]['d_tpump'])
    
    dft[j]['tota_unc_trm'] = \
        np.sqrt(dft[j]['d_im_bkg_trm'] + dft[j]['d_pfe_hum'] + dft[j]['d_eta_c_trm'] + dft[j]['d_eta_a'] + dft[j]['d_tpump'])
    
    ##############################3
    
    dft[j]['d_im_bkg'] = np.sqrt((dft[j]['dI'] ** 2 + dft[j]['dib1'] ** 2) / ((dft[j]['IM'] - dft[j]['iB0']) ** 2))
    dft[j]['d_im_bkg_trm'] = np.sqrt((dft[j]['dI'] ** 2 + dft[j]['dib1'] ** 2 + dft[j]['dbeta'] ** 2) / \
                                     ((dft[j]['IM'] - dft[j]['iB0'] - betac) ** 2))
    # b
    dft[j]['d_pfe_hum'] = np.sqrt(((dft[j]['unc_Cpf_kom'] / dft[j]['Cpf_kom']) ** 2) + \
                                  ((dft[j]['dPhim'] * dft[j]['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2)))
    # c
    dft[j]['d_eta_c'] = (dft[j]['deta_c'] / dft[j]['eta_c'])
    dft[j]['d_eta_c_trm'] = np.sqrt(((ac_err**2 + (np.log10(dft[j]['Pair']) * bc_err)**2)/\
                                            (1 + ac + bc * np.log10(dft[j]['Pair']))**2) + opm_err**2)
    
    
    # d
    dft[j]['d_eta_a'] = (dft[j]['deta_a'] / dft[j]['eta_a'])
    # e
    dft[j]['d_tpump'] = (dft[j]['dtpump'] / dft[j]['Tpump_cor'])
    ################################################################################################################
    # dft = df[(df.Sim == 185) & (df.Team ==5)]
    
    ###plotting
    
    fig = plt.figure(figsize=(15, 12))
    # plt.suptitle("GridSpec Inside GridSpec")
    plt.suptitle(ptitle, fontsize=16)
    
    # gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 2])
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 2])
    gs.update(wspace=0.0005, hspace=0.05)
    
    xrtitle = 'Relative uncertainty [%]'
    
    ax0 = plt.subplot(gs[0])
    plt.yscale('log')
    plt.ylim([1000, 5])
    plt.xlim([-0.5, 25])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlabel(xrtitle, fontsize=16)
    
    plt.plot(dft[j]['tota_unc'] * 100, dft[j]['Pair'], color=cbl[0], label=f'Total uncertainty', linestyle='None',
             marker='p', markersize=1)
    plt.plot(dft[j]['d_im_bkg'] * 100, dft[j]['Pair'], color=cbl[1], label=f'Current, Bkg uncertainty',
             linestyle='None', marker='p', markersize=1)
    plt.plot(dft[j]['d_pfe_hum'] * 100, dft[j]['Pair'], color=cbl[2], label=f'Pump flow rate, efficiency uncertainty ',
             linestyle='None', marker='p', markersize=1)
    plt.plot(dft[j]['d_eta_c'] * 100, dft[j]['Pair'], color=cbl[3], label=f'Conversion uncertainty ', linestyle='None',
             marker='p', markersize=1)
    plt.plot(dft[j]['d_eta_a'] * 100, dft[j]['Pair'], color=cbl[4], label=f'Absorbtion uncertainty ', linestyle='None',
             marker='p', markersize=1)
    plt.plot(dft[j]['d_tpump'] * 100, dft[j]['Pair'], color=cbl[5], label=f'Pump temp. uncertainty ', linestyle='None',
             marker='p', markersize=1)
    
    ax0.legend(loc='upper right', frameon=True, fontsize='large', markerscale=3)
    
    ax1 = plt.subplot(gs[1])
    plt.yscale('log')
    plt.ylim([1000, 5])
    ax1.set_yticklabels([])
    plt.xticks(fontsize=16)
    plt.xlabel('Ozone partial pressure', fontsize=16)
    plt.plot(dft[j]['PO3_dqa'], dft[j]['Pair'], color=cbl[3], label=f'Ozone Partial Pressure -DQA')
    plt.plot(dft[j]['PO3'], dft[j]['Pair'], color=cbl[1], label=f'Ozone Partial Pressure ')
    ax1.legend(loc='upper left', frameon=True, fontsize='large', markerscale=3)
    
    plt.savefig(
        f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v1/Total_Unc/{year}_conventiona_{sp}.png')
    
    plt.show()
    
    ###fig 2 trm
    fig = plt.figure(figsize=(15, 12))
    # plt.suptitle("GridSpec Inside GridSpec")
    plt.suptitle(ptitle, fontsize=16)
    
    # gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 2])
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 2])
    gs.update(wspace=0.0005, hspace=0.05)
    
    xrtitle = 'Relative uncertainty [%]'
    
    ax0 = plt.subplot(gs[0])
    plt.yscale('log')
    plt.ylim([1000, 5])
    plt.xlim([-0.5, 25])
    
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlabel(xrtitle, fontsize=16)
    
    plt.plot(dft[j]['tota_unc_trm'] * 100, dft[j]['Pair'], color=cbl[0], label=f'Total uncertainty ', linestyle='None',
             marker='p', markersize=1)
    plt.plot(dft[j]['d_im_bkg_trm'] * 100, dft[j]['Pair'], color=cbl[1], label=f'Current, Bkg uncertainty',
             linestyle='None', marker='p', markersize=1)
    plt.plot(dft[j]['d_pfe_hum'] * 100, dft[j]['Pair'], color=cbl[2], label=f'Pump flow rate, efficiency uncertainty ',
             linestyle='None', marker='p', markersize=1)
    plt.plot(dft[j]['d_eta_c_trm'] * 100, dft[j]['Pair'], color=cbl[3], label=f'Conversion uncertainty ',
             linestyle='None',
             marker='p', markersize=1)
    plt.plot(dft[j]['d_eta_a'] * 100, dft[j]['Pair'], color=cbl[4], label=f'Absorbtion uncertainty ', linestyle='None',
             marker='p', markersize=1)
    plt.plot(dft[j]['d_tpump'] * 100, dft[j]['Pair'], color=cbl[5], label=f'Pump temp. uncertainty ', linestyle='None',
             marker='p', markersize=1)
    
    ax0.legend(loc='upper right', frameon=True, fontsize='large', markerscale=3)
    
    ax1 = plt.subplot(gs[1])
    plt.yscale('log')
    plt.ylim([1000, 5])
    ax1.set_yticklabels([])
    plt.xticks(fontsize=16)
    plt.xlabel('Ozone partial pressure', fontsize=16)
    # plt.plot(dft[j]['PO3_dqa'], dft[j]['Pair'], color=cbl[3], label=f'Ozone Partial Pressure -DQA')
    plt.plot(dft[j]['PO3_trrm'], dft[j]['Pair'], color=cbl[1], label=f'TRRM Ozone Partial Pressure ')
    ax1.legend(loc='upper left', frameon=True, fontsize='large', markerscale=3)
    
    plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v1/Total_Unc/{year}_TRRM_{sp}.png')
    
    plt.show()
