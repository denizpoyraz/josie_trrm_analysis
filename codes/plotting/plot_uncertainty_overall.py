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
from analyse_functions import Calc_average_profile_pressure, calc_average_df_pressure, set_columns_nopair_dependence
from plotting_functions import filter_rdif, filter_rdif_all

df17c = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_deconv_2023paper.csv", low_memory=False)
df17c = df17c[df17c.iB2 > -9]
df09c = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/"
                    "Josie0910_deconv_2023_unitedpaper.csv", low_memory=False)


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
dfa = dfa[dfa.iB0 < 1]

# year = '2009,2010,2017'
# year = '0910'
# tyear = '2009/2010'
# year = '2017'
# tyear = '2017'
year = '0910'
tyear = '0910'
df = dfa[dfa.Year == year]
#
prof = filter_rdif_all(df)
# # prof = [profEN0505, profEN1010, profEN1001, profSP0505, profSP1010, profSP1001]
beta_l = [beta_en0505,beta_en1010,beta_1001, beta_sp0505,beta_sp1010, beta_1001]
#ilist = [0,1,3,4]0910
#ilist = [0,2,4,5]2017
i = 4
betac = beta_l[i]
dv = 100
ac = a[i]/dv
bc = b[i]/dv
ac_err = a_err[i]/dv
bc_err = b_err[i]/dv
#
dfi = prof[i]



dfi['PO3_dqa'] = 0.043085 * dfi['Tpump_cor'] * (dfi['IM'] - dfi['iB0']) / \
                    (1 * dfi['PFcor_kom'])
if year == '2017':
    dfi['PO3_dqa'] = 0.043085 * dfi['Tpump_cor'] * (dfi['IM'] - dfi['iB2']) / \
                     (1 * dfi['PFcor_kom'])
    if (i == 2) | (i == 5):
        dfi['PO3_dqa'] = 0.043085 * dfi['Tpump_cor'] * (dfi['IM'] - dfi['iB2']) / \
                         (1 * dfi['PFcor_jma'])


dfi['PO3_trrm'] = 0.043085 * dfi['Tpump_cor'] * (dfi['Ifast_minib0_deconv_sm10']) / \
                     (1 * dfi['PFcor_jma'])


if year == '0910':
    dfi['I_slow_conv'] = dfi['I_slow_conv_ib1_decay']

    # column_list = ['Pair','Tsim','IM','Ifast_minib0_deconv_sm10','I_slow_conv','TPext', 'Tpump_cor','PO3','PO3_dqa',
    #                'PO3_trrm','Cpf_kom', 'Cpf_jma','unc_Cpf_kom','unc_Cpf_jma']
if year == '2017':
    dfi['I_slow_conv'] = dfi['Islow_conv']


column_list = ['Pair','Tsim','IM','Ifast_minib0_deconv_sm10','I_slow_conv','TPext', 'Tpump_cor','PO3','PO3_dqa',
                   'PO3_trrm','Cpf_kom', 'Cpf_jma','unc_Cpf_kom','unc_Cpf_jma']

yrefd = [1000, 850, 700, 550, 400, 350, 300, 200, 150, 100, 75, 50, 35,30, 25, 15,9, 8, 6]

df = calc_average_df_pressure(dfi, column_list, yrefd)
nop_columns = ['PFcor', 'iB0', 'iB1', 'iB2']
df = set_columns_nopair_dependence(dfi, df,  nop_columns)

df['a'] = ac
df['b'] = bc
df['a_err'] = ac_err
df['b_err'] = bc_err

df['dI'] = 0
df.loc[df.IM < 1, 'dI'] = 0.005
df.loc[df.IM >= 1, 'dI'] = 0.5 / 100 * df.loc[df.IM > 1, 'IM']
df['dib1'] = 0.02
df['cPL'] = 0.007
df['dcPL'] = 0.002
unc_cPL = 0.002
df['cPH'] = 0.02
df['dcPH'] = 0.002
unc_cPH = 0.002
if year == '2017':
    df['cPH'] = 0.03
    df['dcPH'] = 0.003
    unc_cPH = 0.003

opm_err = 0.02

df['eta_c'] = 1
df['deta_c'] = 0.03
df['eta_a'] = 1
df['deta_a'] = 0.01
df['dtpump'] = 0.7

df['dbeta'] = 0.005

df['dPhim'] = 0.01
df['Phip_ground'] = df['PFcor']
df['unc_Phip_ground'] = df['Phip_ground'] * np.sqrt(
    (df['dPhim']) ** 2 + (unc_cPL) ** 2 + (unc_cPH) ** 2)

df['Phip_cor_kom'], df['unc_Phip_cor_kom'] = return_phipcor(df, 'Phip_ground', 'unc_Phip_ground',
                                                                    'Cpf_kom', 'unc_Cpf_kom')
df['Phip_cor_jma'], df['unc_Phip_cor_jma'] = return_phipcor(df, 'Phip_ground', 'unc_Phip_ground',
                                                                    'Cpf_jma', 'unc_Cpf_jma')


# a
df['d_im_bkg'] = (df['dI'] ** 2 + df['dib1'] ** 2) / ((df['IM'] - df['iB0']) ** 2)
if year == '2017':
    df['d_im_bkg'] = (df['dI'] ** 2 + df['dib1'] ** 2) / ((df['IM'] - df['iB2']) ** 2)

df['d_im_bkg_trm'] = (df['dI'] ** 2 + df['dib1'] ** 2 + df['dbeta'] ** 2) / \
                         ((df['IM'] - df['iB0'] - betac) ** 2)

# b
df['d_pfe_hum'] = ((df['unc_Cpf_kom'] / df['Cpf_kom']) ** 2) + \
                      ((df['dPhim'] * df['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2))
# c

df['d_eta_c'] = (df['deta_c'] / df['eta_c']) ** 2
df['d_eta_c_trm'] = ((ac_err ** 2 + (np.log10(df['Pair']) * bc_err) ** 2) / (
            1 + ac + bc * np.log10(df['Pair'])) ** 2) + opm_err ** 2
# d
df['d_eta_a'] = (df['deta_a'] / df['eta_a']) ** 2
# e
df['d_tpump'] = (df['dtpump'] / df['Tpump_cor']) ** 2

df['tota_unc'] = \
    np.sqrt(df['d_im_bkg'] + df['d_pfe_hum'] + df['d_eta_c'] + df['d_eta_a'] + df['d_tpump'])

df['tota_unc_trm'] = \
    np.sqrt(df['d_im_bkg_trm'] + df['d_pfe_hum'] + df['d_eta_c_trm'] + df['d_eta_a'] + df[
        'd_tpump'])

##############################3

df['d_im_bkg'] = np.sqrt((df['dI'] ** 2 + df['dib1'] ** 2) / ((df['IM'] - df['iB0']) ** 2))
if year == '2017':
    df['d_im_bkg'] = np.sqrt((df['dI'] ** 2 + df['dib1'] ** 2) / ((df['IM'] - df['iB2']) ** 2))

df['d_im_bkg_trm'] = np.sqrt((df['dI'] ** 2 + df['dib1'] ** 2 + df['dbeta'] ** 2) / \
                                 ((df['IM'] - df['iB0'] - betac) ** 2))
# b
df['d_pfe_hum'] = np.sqrt(((df['unc_Cpf_kom'] / df['Cpf_kom']) ** 2) + \
                              ((df['dPhim'] * df['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2)))
# c
df['d_eta_c'] = (df['deta_c'] / df['eta_c'])
df['d_eta_c_trm'] = np.sqrt(((ac_err ** 2 + (np.log10(df['Pair']) * bc_err) ** 2) / (1 + ac + bc * np.log10(df['Pair'])) ** 2) + opm_err ** 2)

# d
df['d_eta_a'] = (df['deta_a'] / df['eta_a'])
# e
df['d_tpump'] = (df['dtpump'] / df['Tpump_cor'])
################################################################################################################
# dft = df[(df.Sim == 185) & (df.Team ==5)]

print(list(df))
df.to_csv(f'/home/poyraden/Analysis/JOSIEfiles/Proccessed/Unc_Josie{year}_{labellist[i]}.csv')
df.to_excel(f'/home/poyraden/Analysis/JOSIEfiles/Proccessed/Unc_Josie{year}_{labellist[i]}.xlsx')

###plotting
trrm_name = f'upd_{year}_TRRM_{labellist[i]}.png'
con_name = f'upd_{year}_conventional_{labellist[i]}.png'

size_label =20
size_title =22

title = f'{tyear} {labellist[i]}'
fig = plt.figure(figsize=(15, 12))
# plt.suptitle("GridSpec Inside GridSpec")
plt.suptitle(title, fontsize=size_title, y = 0.93)
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 2])
# gs.update(wspace=0.0005, hspace=0.05)
gs.update(wspace=0.05, hspace=0.05)


xrtitle = 'Relative uncertainty [%]'
xptitle = 'Ozone partial pressure [mPa]'
ytitle = 'Pressure [hPa]'

ax0 = plt.subplot(gs[0])
plt.yscale('log')
plt.ylim([1000, 5])
plt.xlim([-0.05, 8])
if (year == '2017'):
    plt.xlim([-0.05, 15])

plt.yticks(fontsize=size_label)
plt.xticks(fontsize=size_label)
plt.xlabel(xrtitle, fontsize=size_label)
plt.ylabel(ytitle, fontsize=size_label)
# ax0.set_xticklabels([0,2,4,6])

ax0.yaxis.set_major_formatter(ScalarFormatter())
ax0.tick_params(which='major', width=2)
ax0.tick_params(which='minor', width=2)

plt.plot(df['tota_unc'] * 100, df['Pair'], color=cbl[0], label=f'Total uncertainty', linewidth=2)
plt.plot(df['d_im_bkg'] * 100, df['Pair'], color=cbl[1], label=f'Current, Bkg uncertainty',
         linewidth=2)
plt.plot(df['d_pfe_hum'] * 100, df['Pair'], color=cbl[2], label=f'Pump flow rate, efficiency uncertainty ',
         linewidth=2)
plt.plot(df['d_eta_c'] * 100, df['Pair'], color=cbl[3], label=f'Conversion uncertainty ', linewidth=2)
plt.plot(df['d_eta_a'] * 100, df['Pair'], color=cbl[4], label=f'Absorbtion uncertainty ', linewidth=2)
plt.plot(df['d_tpump'] * 100, df['Pair'], color=cbl[5], label=f'Pump temp. uncertainty ', linewidth=2)

ax0.legend(loc='upper right', frameon=True, fontsize='x-large', markerscale=1,  handletextpad=0.1)

ax1 = plt.subplot(gs[1])
plt.yscale('log')
ax1.yaxis.set_major_formatter(ScalarFormatter())
ax1.tick_params(which='major', width=2)
ax1.tick_params(which='minor', width=2)
plt.ylim([1000, 5])

ax1.set_yticklabels([])
plt.xticks(fontsize=size_label)
plt.xlabel(xptitle, fontsize=size_label)
plt.plot(df['PO3_dqa'], df['Pair'], color=cbl[i], label=f'Ozone Partial Pressure', linewidth=3)
ax1.legend(loc='upper left', frameon=True, fontsize='x-large', markerscale=3)
ax1.xaxis.set_major_formatter(ScalarFormatter())
plt.xticks(np.arange(0,21, 5))


plt.savefig(
    f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v1/Total_Unc/{con_name}')

plt.show()

###fig 2 trm
fig = plt.figure(figsize=(15, 12))
# plt.suptitle("GridSpec Inside GridSpec")
plt.suptitle(title, fontsize=size_title, y = 0.93)
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 2])
gs.update(wspace=0.05, hspace=0.05)

xrtitle = 'Relative uncertainty [%]'

ax0 = plt.subplot(gs[0])
plt.yscale('log')
ax0.yaxis.set_major_formatter(ScalarFormatter())
ax0.tick_params(which='major', width=2)
ax0.tick_params(which='minor', width=2)
plt.ylabel(ytitle, fontsize=size_label)

plt.ylim([1000, 5])
plt.xlim([-0.05, 8])
if (year == '2017'):
    plt.xlim([-0.05, 15])

plt.yticks(fontsize=size_label)
plt.xticks(fontsize=size_label)
plt.xlabel(xrtitle, fontsize=size_label)

plt.plot(df['tota_unc_trm'] * 100, df['Pair'], color=cbl[0], label=f'Total uncertainty ', linewidth=2)
plt.plot(df['d_im_bkg_trm'] * 100, df['Pair'], color=cbl[1], label=f'Current, Bkg uncertainty',
         linewidth=2)
plt.plot(df['d_pfe_hum'] * 100, df['Pair'], color=cbl[2], label=f'Pump flow rate, efficiency uncertainty ',
         linewidth=2)
plt.plot(df['d_eta_c_trm'] * 100, df['Pair'], color=cbl[3], label=f'Conversion uncertainty ',
         linewidth=2)
plt.plot(df['d_eta_a'] * 100, df['Pair'], color=cbl[4], label=f'Absorbtion uncertainty ', linewidth=2)
plt.plot(df['d_tpump'] * 100, df['Pair'], color=cbl[5], label=f'Pump temp. uncertainty ', linewidth=2)

ax0.legend(loc='upper right', frameon=True, fontsize='x-large', markerscale=1,  handletextpad=0.1)

ax1 = plt.subplot(gs[1])
plt.yscale('log')
ax1.yaxis.set_major_formatter(ScalarFormatter())
ax1.tick_params(which='major', width=2)
ax1.tick_params(which='minor', width=2)
plt.ylim([1000, 5])
ax1.set_yticklabels([])
plt.xticks(fontsize=size_label)
plt.xlabel(xptitle, fontsize=size_label)
plt.plot(df['PO3_trrm'], df['Pair'], color=cbl[i], label=f'TRRM Ozone Partial Pressure', linewidth=3)
ax1.legend(loc='upper left', frameon=True, fontsize='x-large', markerscale=3)
plt.xticks(np.arange(0,21, 5))

plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v1/Total_Unc/{trrm_name}')

plt.show()
