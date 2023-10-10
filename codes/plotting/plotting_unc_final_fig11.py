# Libraries
# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter, MultipleLocator, FormatStrFormatter

from analyse_functions import calc_average_df_pressure, set_columns_nopair_dependence, cal_dif
from constant_variables import *
from data_cuts import cuts0910, cuts2017
from homogenization_functions import return_phipcor
from plotting_functions import filter_rdif_all

bool_sm_vh = True
bool_gs = False
if bool_sm_vh:
    pre = '_sm_hv'
else:
    pre = ''

df17c = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_deconv_2023paper_ib2{pre}.csv",
                    low_memory=False)
df17c = df17c[df17c.iB2 > -9]
df09c = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/"
                    f"Josie0910_deconv_2023_unitedpaper{pre}.csv", low_memory=False)

# df09c['Ifast_minib0_deconv_ib1_decay_sm10'] = df09c['Ifast_minib0_deconv_ib1_decay'].rolling(window=5,center=True).mean()
df09c['Ifast_minib0_deconv_sm10'] = df09c['Ifast_minib0_deconv_ib1_decay'].rolling(window=5, center=True).mean()
df17c['Ifast_minib0_deconv_sm10'] = df17c['Ifast_minib0_deconv'].rolling(window=5, center=True).mean()
if bool_sm_vh:
    df09c['Ifast_minib0_deconv_sm10'] = df09c['Ifast_minib0_deconv_ib1_decay']
    df17c['Ifast_minib0_deconv_sm10'] = df17c['Ifast_minib0_deconv']

df09c['Islow_conv'] = df09c['I_slow_conv_ib1_decay']
df09c['Ifast_minib0_deconv'] = df09c['Ifast_minib0_deconv_ib1_decay']
df09c['Ifast_minib0'] = df09c['Ifast_minib0_ib1_decay']

df09 = cuts0910(df09c)
df17 = cuts2017(df17c)

df09['Year'] = '0910'
df17['Year'] = '2017'

dfa = pd.concat([df09, df17], ignore_index=True)
dfa = dfa[dfa.iB0 < 1]

year = '2009,2010,2017'
year = '0910'
tyear = '2009/2010'
# year = '2017'
# tyear = '2017'

df = dfa[dfa.Year == year]

#
prof = filter_rdif_all(df)
# # prof = [profEN0505, profEN1010, profEN1001, profSP0505, profSP1010, profSP1001]
beta_l = [beta_en0505, beta_en1010, beta_1001, beta_sp0505, beta_sp1010, beta_1001]
# 0910
ilist = [0, 1, 3, 4]

size_label = 28
size_title = 32
size_tick = 24
size_legend = 24
# size_label = 22
# size_title = 30
# size_tick = 22
# size_legend = 20
ilist = [0, 1, 3, 4]
one = '(a)'
two = '(b)'
three = '(c)'
# i=4
# j=0
i=3
j=1
if year == '2017':
    ilist = [0, 2, 4, 5]
    one = '(d)'
    two = '(e)'
    three = '(f)'
    i=5
    j=2
    # i=4
    # j=0

betac = beta_l[i]
dv = 100
ac = a[i] / dv
bc = b[i] / dv
ac_err = a_err[i] / dv
bc_err = b_err[i] / dv
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
    dfi['Islow_conv'] = dfi['I_slow_conv_ib1_decay']

if year == '2017':
    dfi['Islow_conv'] = dfi['Islow_conv']

if year == '0910':
    slist = ['PO3_dqa', 'PO3_OPM', 'PO3_trrm', 'PO3', 'IM', 'Ifast_minib0_deconv_sm10', 'Islow_conv',
             'Ifast_minib0_deconv', 'Ifast_minib0']
    clist = ['PO3_dqac', 'PO3_OPMc', 'PO3_trrmc', 'PO3', 'IMc', 'Ifast_minib0_deconv_sm10c', 'Islow_convc',
             'Ifast_minib0_deconvc', 'Ifast_minib0c']
    vlist = ['PFcor', 'iB0', 'iB1', 'iB2', 'Pair', 'ENSCI', 'Sol', 'Buf', 'Sim', 'Team',
             'Tsim', 'TPext', 'Tpump_cor', 'Cpf_kom', 'Cpf_jma', 'unc_Cpf_kom', 'unc_Cpf_jma', 'PFcor_kom',
             'PFcor_jma']

    for ij in range(len(slist)):
        dfi[clist[ij]] = dfi[slist[ij]]
        dfi.loc[(dfi.Pair < 500) & (dfi.Pair > 350) & (dfi.ENSCI == 1), clist[ij]] = np.nan
        dfi.loc[(dfi.Pair < 600) & (dfi.Pair > 300) & (dfi.ENSCI == 0), clist[ij]] = np.nan
        dfi.loc[(dfi.Pair < 120) & (dfi.Pair > 57), clist[ij]] = np.nan
        dfi.loc[(dfi.Pair < 30) & (dfi.Pair > 7), clist[ij]] = np.nan

    dfsi = dfi[clist].interpolate()
    for v in vlist:
        dfsi[v] = dfi[v]

    dfi = dfsi.copy()
    for ij in range(len(slist)):
        dfi[slist[ij]] = 0
        dfi[slist[ij]] = dfsi[clist[ij]]

column_list = ['Pair', 'Tsim', 'IM', 'Ifast_minib0_deconv_sm10', 'Islow_conv', 'TPext', 'Tpump_cor', 'PO3',
               'PO3_dqa',
               'PO3_trrm', 'Cpf_kom', 'Cpf_jma', 'unc_Cpf_kom', 'unc_Cpf_jma', 'PO3_OPM', 'Ifast_minib0_deconv',
               'Ifast_minib0']

dffi = calc_average_df_pressure(dfi, column_list, yref)
nop_columns = ['PFcor', 'iB0', 'iB1', 'iB2', 'PFcor_kom', 'PFcor_jma']
dffi = set_columns_nopair_dependence(dfi, dffi, nop_columns)

dffi['ratio_2'] = -dffi['iB0'] / dffi['IM'] * 100
dffi['ratio_3'] = -dffi['Islow_conv'] / dffi['IM'] * 100
dffi['ratio_6'] = (dffi['Ifast_minib0_deconv'] - dffi['Ifast_minib0']) / dffi['IM'] * 100
dffi['ratio_7'] = (dffi['Cpf_jma'] - 1) * 100

dffi['ratio_8'] = dffi['ratio_2'] + dffi['ratio_3'] + dffi['ratio_6'] + dffi['ratio_7']

dffi['a'] = ac
dffi['b'] = bc
dffi['a_err'] = ac_err
dffi['b_err'] = bc_err

dffi['dI'] = 0
dffi.loc[dffi.IM < 1, 'dI'] = 0.005
dffi.loc[dffi.IM >= 1, 'dI'] = 0.5 / 100 * dffi.loc[dffi.IM > 1, 'IM']
dffi['dib1'] = 0.02
dffi['cPL'] = 0.007
dffi['dcPL'] = 0.002
unc_cPL = 0.002
dffi['cPH'] = 0.02
dffi['dcPH'] = 0.002
unc_cPH = 0.002
if year == '2017':
    dffi['cPH'] = 0.03
    dffi['dcPH'] = 0.003
    unc_cPH = 0.003

opm_err = 0.02

dffi['eta_c'] = 1
dffi['deta_c'] = 0.03
dffi['eta_a'] = 1
dffi['deta_a'] = 0.01
dffi['dtpump'] = 0.7

dffi['dbeta'] = 0.005

dffi['dPhim'] = 0.01
dffi['Phip_ground'] = dffi['PFcor']
dffi['unc_Phip_ground'] = dffi['Phip_ground'] * np.sqrt(
    (dffi['dPhim']) ** 2 + (unc_cPL) ** 2 + (unc_cPH) ** 2)

dffi['Phip_cor_kom'], dffi['unc_Phip_cor_kom'] = return_phipcor(dffi, 'Phip_ground', 'unc_Phip_ground',
                                                                  'Cpf_kom', 'unc_Cpf_kom')
dffi['Phip_cor_jma'], dffi['unc_Phip_cor_jma'] = return_phipcor(dffi, 'Phip_ground', 'unc_Phip_ground',
                                                                  'Cpf_jma', 'unc_Cpf_jma')

# a
dffi['d_im_bkg'] = (dffi['dI'] ** 2 + dffi['dib1'] ** 2) / ((dffi['IM'] - dffi['iB0']) ** 2)
if year == '2017':
    dffi['d_im_bkg'] = (dffi['dI'] ** 2 + dffi['dib1'] ** 2) / ((dffi['IM'] - dffi['iB2']) ** 2)

dffi['d_im_bkg_trm'] = (dffi['dI'] ** 2 + dffi['dib1'] ** 2 + dffi['dbeta'] ** 2) / \
                        ((dffi['IM'] - dffi['iB0'] - betac) ** 2)

# b
dffi['d_pfe_hum'] = ((dffi['unc_Cpf_kom'] / dffi['Cpf_kom']) ** 2) + \
                     ((dffi['dPhim'] * dffi['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2))
dffi['d_pfe_hum_trrm'] = ((dffi['unc_Cpf_jma'] / dffi['Cpf_jma']) ** 2) + \
                          ((dffi['dPhim'] * dffi['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2))
# c

dffi['d_eta_c'] = (dffi['deta_c'] / dffi['eta_c']) ** 2
dffi['d_eta_c_trm'] = ((ac_err ** 2 + (np.log10(dffi['Pair']) * bc_err) ** 2) / (
        1 + ac + bc * np.log10(dffi['Pair'])) ** 2) + opm_err ** 2
# d
dffi['d_eta_a'] = (dffi['deta_a'] / dffi['eta_a']) ** 2
# e
dffi['d_tpump'] = (dffi['dtpump'] / dffi['Tpump_cor']) ** 2

dffi['tota_unc'] = \
    np.sqrt(dffi['d_im_bkg'] + dffi['d_pfe_hum'] + dffi['d_eta_c'] + dffi['d_eta_a'] + dffi['d_tpump'])

dffi['tota_unc_trm'] = \
    np.sqrt(dffi['d_im_bkg_trm'] + dffi['d_pfe_hum_trrm'] + dffi['d_eta_c_trm'] + dffi['d_eta_a'] + dffi[
        'd_tpump'])

##############################3

dffi['d_im_bkg'] = np.sqrt((dffi['dI'] ** 2 + dffi['dib1'] ** 2) / ((dffi['IM'] - dffi['iB0']) ** 2))
if year == '2017':
    dffi['d_im_bkg'] = np.sqrt((dffi['dI'] ** 2 + dffi['dib1'] ** 2) / ((dffi['IM'] - dffi['iB2']) ** 2))

dffi['d_im_bkg_trm'] = np.sqrt((dffi['dI'] ** 2 + dffi['dib1'] ** 2 + dffi['dbeta'] ** 2) / \
                                ((dffi['IM'] - dffi['iB0'] - betac) ** 2))
# b
dffi['d_pfe_hum'] = np.sqrt(((dffi['unc_Cpf_kom'] / dffi['Cpf_kom']) ** 2) + \
                             ((dffi['dPhim'] * dffi['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2)))
dffi['d_pfe_hum_trrm'] = np.sqrt(((dffi['unc_Cpf_jma'] / dffi['Cpf_jma']) ** 2) + \
                                  ((dffi['dPhim'] * dffi['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2)))
# c
dffi['d_eta_c'] = (dffi['deta_c'] / dffi['eta_c'])
dffi['d_eta_c_trm'] = np.sqrt(((ac_err ** 2 + (np.log10(dffi['Pair']) * bc_err) ** 2) / (
        1 + ac + bc * np.log10(dffi['Pair'])) ** 2) + opm_err ** 2)

# d
dffi['d_eta_a'] = (dffi['deta_a'] / dffi['eta_a'])
# e
dffi['d_tpump'] = (dffi['dtpump'] / dffi['Tpump_cor'])
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
betac = beta_l[j]
dv = 100
ac = a[j] / dv
bc = b[j] / dv
ac_err = a_err[j] / dv
bc_err = b_err[j] / dv
#
dfj = prof[j]

dfj['PO3_dqa'] = 0.043085 * dfj['Tpump_cor'] * (dfj['IM'] - dfj['iB0']) / \
                    (1 * dfj['PFcor_kom'])
if year == '2017':
    dfj['PO3_dqa'] = 0.043085 * dfj['Tpump_cor'] * (dfj['IM'] - dfj['iB2']) / \
                        (1 * dfj['PFcor_kom'])
    if (i == 2) | (i == 5):
        dfj['PO3_dqa'] = 0.043085 * dfj['Tpump_cor'] * (dfj['IM'] - dfj['iB2']) / \
                            (1 * dfj['PFcor_jma'])

dfj['PO3_trrm'] = 0.043085 * dfj['Tpump_cor'] * (dfj['Ifast_minib0_deconv_sm10']) / \
                     (1 * dfj['PFcor_jma'])

if year == '0910':
    dfj['Islow_conv'] = dfj['I_slow_conv_ib1_decay']

if year == '2017':
    dfj['Islow_conv'] = dfj['Islow_conv']

if year == '0910':
    slist = ['PO3_dqa', 'PO3_OPM', 'PO3_trrm', 'PO3', 'IM', 'Ifast_minib0_deconv_sm10', 'Islow_conv',
             'Ifast_minib0_deconv', 'Ifast_minib0']
    clist = ['PO3_dqac', 'PO3_OPMc', 'PO3_trrmc', 'PO3', 'IMc', 'Ifast_minib0_deconv_sm10c', 'Islow_convc',
             'Ifast_minib0_deconvc', 'Ifast_minib0c']
    vlist = ['PFcor', 'iB0', 'iB1', 'iB2', 'Pair', 'ENSCI', 'Sol', 'Buf', 'Sim', 'Team',
             'Tsim', 'TPext', 'Tpump_cor', 'Cpf_kom', 'Cpf_jma', 'unc_Cpf_kom', 'unc_Cpf_jma', 'PFcor_kom',
             'PFcor_jma']

    for k in range(len(slist)):
        dfj[clist[k]] = dfj[slist[k]]
        dfj.loc[(dfj.Pair < 500) & (dfj.Pair > 350) & (dfj.ENSCI == 1), clist[k]] = np.nan
        dfj.loc[(dfj.Pair < 600) & (dfj.Pair > 300) & (dfj.ENSCI == 0), clist[k]] = np.nan
        dfj.loc[(dfj.Pair < 120) & (dfj.Pair > 57), clist[k]] = np.nan
        dfj.loc[(dfj.Pair < 30) & (dfj.Pair > 7), clist[k]] = np.nan

    dfsj = dfj[clist].interpolate()
    for v in vlist:
        dfsj[v] = dfj[v]

    dfj = dfsj.copy()
    for z in range(len(slist)):
        dfj[slist[z]] = 0
        dfj[slist[z]] = dfsj[clist[z]]

column_list = ['Pair', 'Tsim', 'IM', 'Ifast_minib0_deconv_sm10', 'Islow_conv', 'TPext', 'Tpump_cor', 'PO3',
               'PO3_dqa',
               'PO3_trrm', 'Cpf_kom', 'Cpf_jma', 'unc_Cpf_kom', 'unc_Cpf_jma', 'PO3_OPM', 'Ifast_minib0_deconv',
               'Ifast_minib0']

dffj = calc_average_df_pressure(dfj, column_list, yref)
nop_columns = ['PFcor', 'iB0', 'iB1', 'iB2', 'PFcor_kom', 'PFcor_jma']
dffj = set_columns_nopair_dependence(dfj, dffj, nop_columns)

dffj['ratio_2'] = -dffj['iB0'] / dffj['IM'] * 100
dffj['ratio_3'] = -dffj['Islow_conv'] / dffj['IM'] * 100
dffj['ratio_6'] = (dffj['Ifast_minib0_deconv'] - dffj['Ifast_minib0']) / dffj['IM'] * 100
dffj['ratio_7'] = (dffj['Cpf_jma'] - 1) * 100

dffj['ratio_8'] = dffj['ratio_2'] + dffj['ratio_3'] + dffj['ratio_6'] + dffj['ratio_7']

dffj['a'] = ac
dffj['b'] = bc
dffj['a_err'] = ac_err
dffj['b_err'] = bc_err

dffj['dI'] = 0
dffj.loc[dffj.IM < 1, 'dI'] = 0.005
dffj.loc[dffj.IM >= 1, 'dI'] = 0.5 / 100 * dffj.loc[dffj.IM > 1, 'IM']
dffj['dib1'] = 0.02
dffj['cPL'] = 0.007
dffj['dcPL'] = 0.002
unc_cPL = 0.002
dffj['cPH'] = 0.02
dffj['dcPH'] = 0.002
unc_cPH = 0.002
if year == '2017':
    dffj['cPH'] = 0.03
    dffj['dcPH'] = 0.003
    unc_cPH = 0.003

opm_err = 0.02

dffj['eta_c'] = 1
dffj['deta_c'] = 0.03
dffj['eta_a'] = 1
dffj['deta_a'] = 0.01
dffj['dtpump'] = 0.7

dffj['dbeta'] = 0.005

dffj['dPhim'] = 0.01
dffj['Phip_ground'] = dffj['PFcor']
dffj['unc_Phip_ground'] = dffj['Phip_ground'] * np.sqrt(
    (dffj['dPhim']) ** 2 + (unc_cPL) ** 2 + (unc_cPH) ** 2)

dffj['Phip_cor_kom'], dffj['unc_Phip_cor_kom'] = return_phipcor(dffj, 'Phip_ground', 'unc_Phip_ground',
                                                                  'Cpf_kom', 'unc_Cpf_kom')
dffj['Phip_cor_jma'], dffj['unc_Phip_cor_jma'] = return_phipcor(dffj, 'Phip_ground', 'unc_Phip_ground',
                                                                  'Cpf_jma', 'unc_Cpf_jma')

# a
dffj['d_im_bkg'] = (dffj['dI'] ** 2 + dffj['dib1'] ** 2) / ((dffj['IM'] - dffj['iB0']) ** 2)
if year == '2017':
    dffj['d_im_bkg'] = (dffj['dI'] ** 2 + dffj['dib1'] ** 2) / ((dffj['IM'] - dffj['iB2']) ** 2)

dffj['d_im_bkg_trm'] = (dffj['dI'] ** 2 + dffj['dib1'] ** 2 + dffj['dbeta'] ** 2) / \
                        ((dffj['IM'] - dffj['iB0'] - betac) ** 2)

# b
dffj['d_pfe_hum'] = ((dffj['unc_Cpf_kom'] / dffj['Cpf_kom']) ** 2) + \
                     ((dffj['dPhim'] * dffj['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2))
dffj['d_pfe_hum_trrm'] = ((dffj['unc_Cpf_jma'] / dffj['Cpf_jma']) ** 2) + \
                          ((dffj['dPhim'] * dffj['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2))
# c

dffj['d_eta_c'] = (dffj['deta_c'] / dffj['eta_c']) ** 2
dffj['d_eta_c_trm'] = ((ac_err ** 2 + (np.log10(dffj['Pair']) * bc_err) ** 2) / (
        1 + ac + bc * np.log10(dffj['Pair'])) ** 2) + opm_err ** 2
# d
dffj['d_eta_a'] = (dffj['deta_a'] / dffj['eta_a']) ** 2
# e
dffj['d_tpump'] = (dffj['dtpump'] / dffj['Tpump_cor']) ** 2

dffj['tota_unc'] = \
    np.sqrt(dffj['d_im_bkg'] + dffj['d_pfe_hum'] + dffj['d_eta_c'] + dffj['d_eta_a'] + dffj['d_tpump'])

dffj['tota_unc_trm'] = \
    np.sqrt(dffj['d_im_bkg_trm'] + dffj['d_pfe_hum_trrm'] + dffj['d_eta_c_trm'] + dffj['d_eta_a'] + dffj[
        'd_tpump'])

##############################3

dffj['d_im_bkg'] = np.sqrt((dffj['dI'] ** 2 + dffj['dib1'] ** 2) / ((dffj['IM'] - dffj['iB0']) ** 2))
if year == '2017':
    dffj['d_im_bkg'] = np.sqrt((dffj['dI'] ** 2 + dffj['dib1'] ** 2) / ((dffj['IM'] - dffj['iB2']) ** 2))

dffj['d_im_bkg_trm'] = np.sqrt((dffj['dI'] ** 2 + dffj['dib1'] ** 2 + dffj['dbeta'] ** 2) / \
                                ((dffj['IM'] - dffj['iB0'] - betac) ** 2))
# b
dffj['d_pfe_hum'] = np.sqrt(((dffj['unc_Cpf_kom'] / dffj['Cpf_kom']) ** 2) + \
                             ((dffj['dPhim'] * dffj['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2)))
dffj['d_pfe_hum_trrm'] = np.sqrt(((dffj['unc_Cpf_jma'] / dffj['Cpf_jma']) ** 2) + \
                                  ((dffj['dPhim'] * dffj['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2)))
# c
dffj['d_eta_c'] = (dffj['deta_c'] / dffj['eta_c'])
dffj['d_eta_c_trm'] = np.sqrt(((ac_err ** 2 + (np.log10(dffj['Pair']) * bc_err) ** 2) / (
        1 + ac + bc * np.log10(dffj['Pair'])) ** 2) + opm_err ** 2)

# d
dffj['d_eta_a'] = (dffj['deta_a'] / dffj['eta_a'])
# e
dffj['d_tpump'] = (dffj['dtpump'] / dffj['Tpump_cor'])
################################################################################################################
################################################################################################################






# plotting

# # prof = [profEN0505, profEN1010, profEN1001, profSP0505, profSP1010, profSP1001]
beta_l = [beta_en0505, beta_en1010, beta_1001, beta_sp0505, beta_sp1010, beta_1001]
# 0910


xrtitle = 'Rel. uncertainty [%]'
xptitle = 'Ozone partial pressure [mPa]'
ytitle = 'Pressure [hPa]'
xptitle2 = 'O3 [mPa]'

###plotting
trrm_name = f'unc_{pre}_{year}_TRRM_{labellist_n[i]}.png'
print(trrm_name)

title1 = f'{tyear} {labellist[i]}'
title2 = f'{tyear} {labellist[j]}'

titleo = f'{tyear}'
###fig 2 trm
fig = plt.figure(figsize=(23, 15))
# plt.suptitle("GridSpec Inside GridSpec")
# plt.suptitle(maintitle, fontsize=size_title, y = 0.93)

gs = gridspec.GridSpec(1, 3, width_ratios=[2.1, 2, 2])
gs.update(wspace=0.0005, hspace=0.05)
ax = plt.subplot(gs[0])

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(1.5)  # change width
plt.yscale('log')
ax.set_title(titleo, fontsize=size_title)
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xlabel(xrtitle, fontsize=size_label)
plt.ylabel(ytitle, fontsize=size_label, labelpad=-15)
plt.xticks(fontsize=size_tick)
plt.yticks(fontsize=size_tick)
plt.gca().tick_params(which='major', width=3)
plt.gca().tick_params(which='minor', width=3)
# plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().xaxis.set_tick_params(length=5, which='minor')
plt.gca().yaxis.set_tick_params(length=5, which='minor')
plt.gca().xaxis.set_tick_params(length=10, which='major')
plt.gca().yaxis.set_tick_params(length=10, which='major')
plt.ylim([1000, 5])
plt.xlim([0, 25])

if year == '0910':
    plt.ylim([1000, 5])
# ax.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))
# plt.xticks(np.arange(-0.05, 0.19, 0.05))
ax.set_xticklabels([0, 5,10,15,20,''])

# ax.set_yticklabels([])
# plt.xticks(np.arange(-0.05, 0.19, 0.05))

# plt.ylabel(ytitle, fontsize=size_legend)
plt.yticks(fontsize=size_tick)
plt.ylabel(ytitle, fontsize=size_label)
plt.xticks(fontsize=size_tick)
plt.xlabel(xptitle, fontsize=size_label)
plt.plot(dffi['PO3_OPM'], dffi['Pair'], color=cbl[8], label=f'OPM', linewidth=3)


# ax.legend(loc='lower right', frameon=True, fontsize=size_legend, markerscale=1, handletextpad=0.1, framealpha=0.65)
ax.legend(loc='best', frameon=True, fontsize=size_legend, markerscale=1, handletextpad=0.1, framealpha=0.65)
plt.text(20,900, one,  fontsize=size_title)

ax0 = plt.subplot(gs[1])
plt.yscale('log')
for axis in ['top', 'bottom', 'left', 'right']:
    ax0.spines[axis].set_linewidth(1.5)  # change width
ax0.set_title(title1, fontsize=size_title)
plt.xlabel(xrtitle, fontsize=size_label)
plt.xticks(fontsize=size_tick)
# plt.yticks(fontsize=size_tick)
# plt.grid(True)
plt.gca().tick_params(which='major', width=3)
plt.gca().tick_params(which='minor', width=3)
plt.gca().xaxis.set_tick_params(length=5, which='minor')
plt.gca().yaxis.set_tick_params(length=5, which='minor')
plt.gca().xaxis.set_tick_params(length=10, which='major')
plt.gca().yaxis.set_tick_params(length=10, which='major')
plt.ylim([1000, 5])
plt.xlim([0, 20])
plt.xticks(np.arange(0, 20, 5))

# ax0.set_xticklabels([0, 5,10,15,''])

if year == '0910':
    plt.ylim([1000, 5])

plt.xticks(fontsize=size_tick)
# plt.xlabel('Ratio', fontsize=size_legend)
ax0.set_yticklabels([])

plt.xlabel('Rel. uncertainity [%]', fontsize=size_label)
plt.plot(dffi['d_im_bkg_trm'] * 100, dffi['Pair'], color=cbl[1], label=f'Current, Bkg',
         linewidth=4)
plt.plot(dffi['d_pfe_hum_trrm'] * 100, dffi['Pair'], color=cbl[2], label=f'Pump flow rate eff.',
         linewidth=3)
plt.plot(dffi['d_eta_c_trm'] * 100, dffi['Pair'], color=cbl[3], label=f'Conversion',
         linewidth=4)
plt.plot(dffi['d_eta_a'] * 100, dffi['Pair'], color=cbl[4], label=f'Absorption', linewidth=4)
plt.plot(dffi['d_tpump'] * 100, dffi['Pair'], color=cbl[5], label=f'Pump temp.', linewidth=4)
plt.plot(dffi['tota_unc_trm'] * 100, dffi['Pair'], color=cbl[0], label=f'Total TRC  ', linewidth=4)
plt.plot(dffi['tota_unc'] * 100, dffi['Pair'], color=cbl[0], label=f'Total conventional', linewidth=4, linestyle="-.")


# plt.text(-23,7, two,  fontsize=size_title, bbox={'facecolor': 'white', 'alpha': 0.6})
plt.text(17,900, two,  fontsize=size_title)

ax0.legend(loc='upper right', frameon=True, fontsize=size_legend, markerscale=1, handletextpad=0.1, framealpha=0.65)


######
ax1 = plt.subplot(gs[2])
plt.yscale('log')
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.5)  # change width
ax1.set_title(title2, fontsize=size_title)
plt.xlabel(xrtitle, fontsize=size_label)
plt.xticks(fontsize=size_tick)

plt.gca().tick_params(which='major', width=3)
plt.gca().tick_params(which='minor', width=3)
plt.gca().xaxis.set_tick_params(length=5, which='minor')
plt.gca().yaxis.set_tick_params(length=5, which='minor')
plt.gca().xaxis.set_tick_params(length=10, which='major')
plt.gca().yaxis.set_tick_params(length=10, which='major')
plt.ylim([1000, 5])
plt.xlim([0, 20])
plt.xticks(np.arange(0, 20, 5))

if year == '0910':
    plt.ylim([1000, 5])

# ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))

# plt.yticks(fontsize=size_legend)
# plt.ylabel(ytitle, fontsize=size_legend)
plt.xticks(fontsize=size_tick)
# plt.xlabel('Ratio', fontsize=size_legend)
ax1.set_yticklabels([])

plt.xlabel('Rel. uncertainity [%]', fontsize=size_label)
plt.plot(dffj['d_im_bkg_trm'] * 100, dffj['Pair'], color=cbl[1], label=f'Current, Bkg',
         linewidth=4)
plt.plot(dffj['d_pfe_hum_trrm'] * 100, dffj['Pair'], color=cbl[2], label=f'Pump flow rate eff.',
         linewidth=3)
plt.plot(dffj['d_eta_c_trm'] * 100, dffj['Pair'], color=cbl[3], label=f'Conversion',
         linewidth=4)
plt.plot(dffj['d_eta_a'] * 100, dffj['Pair'], color=cbl[4], label=f'Absorption', linewidth=4)
plt.plot(dffj['d_tpump'] * 100, dffj['Pair'], color=cbl[5], label=f'Pump temp.', linewidth=4)
plt.plot(dffj['tota_unc_trm'] * 100, dffj['Pair'], color=cbl[0], label=f'Total TRC  ', linewidth=4)
plt.plot(dffj['tota_unc'] * 100, dffj['Pair'], color=cbl[0], label=f'Total conventional', linewidth=4, linestyle="-.")
plt.text(17,900, three,  fontsize=size_title)

ax1.legend(loc='upper right', frameon=True, fontsize=size_legend, markerscale=1, handletextpad=0.1, framealpha=0.65)
# plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v5/Total_Unc/fig11_one_{year}')

plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v5/Total_Unc/sup9_one_{year}')
plt.show()
####unc plots
# if not bool_gs:
#     fig, ax1 = plt.subplots(figsize=(15, 22))
# if bool_gs:
#     ax1 = plt.subplot(gs[1])
#
# plt.yscale('log')
# plt.ylim([1000, 5])
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax1.spines[axis].set_linewidth(1.5)  # change width
# ax1.set_title(title, fontsize=size_legend)
# if bool_gs:
#     ax1.set_title('Uncertainty Budget', fontsize=size_legend)
# ax1.yaxis.set_major_formatter(ScalarFormatter())
# ax1.xaxis.set_major_formatter(ScalarFormatter())
#
# ax1.tick_params(which='major', width=3)
# ax1.tick_params(which='minor', width=3)
# # plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
# ax1.xaxis.set_tick_params(length=5, width=3, which='minor')
# ax1.xaxis.set_tick_params(length=10, which='major')
# ax1.yaxis.set_tick_params(length=5, which='minor')
# ax1.yaxis.set_tick_params(length=10, which='major')
# if not bool_gs:
#     # plt.ylabel(ytitle, fontsize=size_legend)
#     ax1.set_yticklabels([])
#
# if bool_gs:
#     ax1.set_yticklabels([])
# ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))
#
# # plt.yticks(fontsize=size_legend)
# plt.xticks(fontsize=size_legend)
# plt.xlabel(xrtitle, fontsize=size_legend)
#
# plt.plot(df['d_im_bkg_trm'] * 100, df['Pair'], color=cbl[1], label=f'Current, Bkg',
#          linewidth=4)
# plt.plot(df['d_pfe_hum_trrm'] * 100, df['Pair'], color=cbl[2], label=f'Pump flow rate eff.',
#          linewidth=3)
# plt.plot(df['d_eta_c_trm'] * 100, df['Pair'], color=cbl[3], label=f'Conversion',
#          linewidth=4)
# plt.plot(df['d_eta_a'] * 100, df['Pair'], color=cbl[4], label=f'Absorption', linewidth=4)
# plt.plot(df['d_tpump'] * 100, df['Pair'], color=cbl[5], label=f'Pump temp.', linewidth=4)
# plt.plot(df['tota_unc_trm'] * 100, df['Pair'], color=cbl[0], label=f'Total TRC  ', linewidth=4)
# plt.plot(df['tota_unc'] * 100, df['Pair'], color=cbl[0], label=f'Total conventional', linewidth=4, linestyle="-.")
#
# ax1.legend(loc='lower right', frameon=True, fontsize=size_legend, markerscale=1, handletextpad=0.1)
# plt.xlim([0, 20])
#
# if not bool_gs:
#     plt.savefig(
#         f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v5/Total_Unc/uncertainity_budget_{trrm_name}')
# if bool_gs:
#     plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v5/Total_Unc/all_{trrm_name}')
#
# # plt.show()
