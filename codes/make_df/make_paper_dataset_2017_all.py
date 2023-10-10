import pandas as pd
import numpy as np
from plotting_functions import filter_rdif_all
from constant_variables import *
from homogenization_functions import return_phipcor

df = pd.read_csv(f'/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_calibrated_sm_hv.csv')
year='2017'

df['PO3_cal'] = (0.043085 * df['Tpump_cor'] * df['I_corrected']) / (df['PFcor_jma'])

ilist = [0,2,4,5]
beta_l = [beta_en0505,beta_en1010,beta_1001, beta_sp0505,beta_sp1010, beta_1001]
prof = filter_rdif_all(df)


for i in ilist:
    betac = beta_l[i]
    dv = 100
    ac = a[i]/dv
    bc = b[i]/dv
    ac_err = a_err[i]/dv
    bc_err = b_err[i]/dv
    #
    dfi = prof[i]

    eta = 1 + a_smhv[i]/100 + b_smhv[i]/100* np.log10(dfi['Pair'])

    dfi['etac'] = eta


    dfi['PO3_dqa'] = 0.043085 * dfi['Tpump_cor'] * (dfi['IM'] - dfi['iB2']) / \
                     (1 * dfi['PFcor_kom'])
    dfi['PF_con'] = dfi['PFcor_kom']
    if (i == 2) | (i == 5):
        dfi['PO3_dqa'] = 0.043085 * dfi['Tpump_cor'] * (dfi['IM'] - dfi['iB2']) / \
                         (1 * dfi['PFcor_jma'])
        dfi['PF_con'] = dfi['PFcor_jma']

    dfi['PO3_trrm'] = 0.043085 * dfi['Tpump_cor'] * (dfi['Ifast_minib0_deconv_sm10']) / \
                         (1 * dfi['PFcor_jma'])


    dfi['a'] = ac
    dfi['b'] = bc
    dfi['a_err'] = ac_err
    dfi['b_err'] = bc_err

    dfi['dI'] = 0
    dfi.loc[dfi.IM < 1, 'dI'] = 0.005
    dfi.loc[dfi.IM >= 1, 'dI'] = 0.5 / 100 * dfi.loc[dfi.IM > 1, 'IM']
    dfi['dib1'] = 0.02
    dfi['cPL'] = 0.007
    dfi['dcPL'] = 0.002
    unc_cPL = 0.002
    dfi['cPH'] = 0.02
    dfi['dcPH'] = 0.002
    unc_cPH = 0.002
    if year == '2017':
        dfi['cPH'] = 0.03
        dfi['dcPH'] = 0.003
        unc_cPH = 0.003

    opm_err = 0.02

    dfi['eta_c'] = 1
    dfi['deta_c'] = 0.03
    dfi['eta_a'] = 1
    dfi['deta_a'] = 0.01
    dfi['dtpump'] = 0.7

    dfi['dbeta'] = 0.005

    dfi['dPhim'] = 0.01
    dfi['Phip_ground'] = dfi['PFcor']
    dfi['unc_Phip_ground'] = dfi['Phip_ground'] * np.sqrt(
        (dfi['dPhim']) ** 2 + (unc_cPL) ** 2 + (unc_cPH) ** 2)

    dfi['Phip_cor_kom'], dfi['unc_Phip_cor_kom'] = return_phipcor(dfi, 'Phip_ground', 'unc_Phip_ground',
                                                                        'Cpf_kom', 'unc_Cpf_kom')
    dfi['Phip_cor_jma'], dfi['unc_Phip_cor_jma'] = return_phipcor(dfi, 'Phip_ground', 'unc_Phip_ground',
                                                                        'Cpf_jma', 'unc_Cpf_jma')

    # a
    dfi['d_im_bkg'] = (dfi['dI'] ** 2 + dfi['dib1'] ** 2) / ((dfi['IM'] - dfi['iB0']) ** 2)
    if year == '2017':
        dfi['d_im_bkg'] = (dfi['dI'] ** 2 + dfi['dib1'] ** 2) / ((dfi['IM'] - dfi['iB2']) ** 2)

    dfi['d_im_bkg_trm'] = (dfi['dI'] ** 2 + dfi['dib1'] ** 2 + dfi['dbeta'] ** 2) / \
                             ((dfi['IM'] - dfi['iB0'] - betac) ** 2)

    # b
    dfi['d_pfe_hum'] = ((dfi['unc_Cpf_kom'] / dfi['Cpf_kom']) ** 2) + \
                          ((dfi['dPhim'] * dfi['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2))
    dfi['d_pfe_hum_trrm'] = ((dfi['unc_Cpf_jma'] / dfi['Cpf_jma']) ** 2) + \
                      ((dfi['dPhim'] * dfi['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2))
    # c

    dfi['d_eta_c'] = (dfi['deta_c'] / dfi['eta_c']) ** 2
    dfi['d_eta_c_trm'] = ((ac_err ** 2 + (np.log10(dfi['Pair']) * bc_err) ** 2) / (
                1 + ac + bc * np.log10(dfi['Pair'])) ** 2) + opm_err ** 2
    # d
    dfi['d_eta_a'] = (dfi['deta_a'] / dfi['eta_a']) ** 2
    # e
    dfi['d_tpump'] = (dfi['dtpump'] / dfi['Tpump_cor']) ** 2

    dfi['tota_unc'] = \
        np.sqrt(dfi['d_im_bkg'] + dfi['d_pfe_hum'] + dfi['d_eta_c'] + dfi['d_eta_a'] + dfi['d_tpump'])

    dfi['tota_unc_trm'] = \
        np.sqrt(dfi['d_im_bkg_trm'] + dfi['d_pfe_hum_trrm'] + dfi['d_eta_c_trm'] + dfi['d_eta_a'] + dfi[
            'd_tpump'])



dfc = pd.concat([prof[0], prof[2], prof[4], prof[5]], ignore_index=True)

print(dfc)
dff = pd.DataFrame()
units = ['sec','','hPa','K','muA','muA','muA','muA','K','K','','','','','','mPa','mPa','mPa','mPa''mPa','mPa','mPa','mPa']
# dft[['Tsim', 'Pair','Tair','IM','I_conv_slow','Ifast_minib0_deconv_sm10']]
dff['Sim_Time'] = dfc['Tsim']
dff['Data_Index'] = dfc['Tsim']
dff['Pair_Pressure_ESC'] = dfc['Pair']
dff['Tair_Pressure_ESC'] = dfc['Tair']
dff['IM_Cell_Measured'] = dfc['IM']
dff['IS_Cell_Slow'] = dfc['I_conv_slow']
dff['IF_Cell_Fast'] = dfc['Ifast_minib0']
dff['IFDS_Cell_Fast_Deconv_Smooth'] = dfc['Ifast_minib0_deconv']
dff['Pump_T_Ext'] = dfc['TPint']
dff['Pump_T_Int'] = dfc['TPext']
dff['Absorption_Eff'] = 1
dff['Pump_Eff_Conv'] = dfc['PF_con']
dff['Conversion_Eff_Conv'] = 1
dff['Pump_Eff_TRCC'] = dfc['PFcor_jma']
dff['Conversion_Eff_TRCC'] = dfc['etac']
dff['PO3_Conv'] = dfc['PO3_dqa']
dff['PO3_TRC'] = dfc['PO3_trrm']
dff['PO3_TRCC'] = dfc['PO3_cal']
dff['PO3_OPM'] = dfc['PO3_OPM']
dff['Unc_PO3_Conv'] = dfc['tota_unc']
dff['Unc_PO3_TRC'] = np.nan
dff['Unc_PO3_TRCC'] = dfc['tota_unc_trm']
dff['Unc_PO3_OPM'] = opm_err
dff['Sim'] = dfc['Sim']
dff['Team'] = dfc['Team']


dff.to_csv('/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_Data_Public.csv')


