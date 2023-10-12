import pandas as pd
import numpy as np
from constant_variables import *
from homogenization_functions import return_phipcor
import math

year = '0910'
bool_pre = True
pre = '_publicdata'
end = 'simulation'
if bool_pre:
    pre = '_all' #for preparation data
    end = "_preparation"


print(f'/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_calibrated_sm_hv{pre}.csv')
df = pd.read_csv(f'/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_calibrated_sm_hv{pre}.csv')
##needed to get some metadata
dfi = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Data_2023paper.csv")

if not bool_pre: df['I_slow_conv'] = df['I_slow_conv_ib1_decay']
if bool_pre:
    df = df[df.TimeTag=='Prep']
    df['I_slow_conv'] = df['I_slow_conv_ib1_decay_all']

df['PO3_cal'] = (0.043085 * df['Tpump_cor'] * df['I_corrected']) / (df['PFcor_jma'])

simlist = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Sim'])
teamlist = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Team'])
sol = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Sol'].tolist())
buff = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Buf'])
ensci = np.asarray(df.drop_duplicates(['Sim', 'Team'])['ENSCI'])

dft = {}

for j in range(len(simlist)):

    betac = 0
    dv = 100
    # labellist = [0'EN-SCI SST0.5', 1'EN-SCI SST1.0', 2'EN-SCI SST0.1',3'SPC SST0.5', 4'SPC SST1.0', 5'SPC SST0.1']

    if (ensci[j] == 1) & (sol[j] == 0.5) & (buff[j] == 0.5):
        betac = beta_en0505
        ac = a_smhv[0] / dv
        bc = b_smhv[0] / dv
        ac_err = a_smhv_err[0] / dv
        bc_err = b_smhv_err[0] / dv
        sst = 'SST0.5/0.5'

    if (ensci[j] == 1) & (sol[j] == 1.0) & (buff[j] == 1.0):
        betac = beta_en1010
        ac = a_smhv[1] / dv
        bc = b_smhv[1] / dv
        ac_err = a_smhv_err[1] / dv
        bc_err = b_smhv_err[1] / dv
        sst = 'SST1.0/1.0'

    if (ensci[j] == 0) & (sol[j] == 0.5) & (buff[j] == 0.5):
        betac = beta_sp0505
        ac = a_smhv[3] / dv
        bc = b_smhv[3] / dv
        ac_err = a_smhv_err[3] / dv
        bc_err = b_smhv_err[3] / dv
        sst = 'SST0.5/0.5'

    if (ensci[j] == 0) & (sol[j] == 1.0) & (buff[j] == 1.0):
        betac = beta_sp1010
        ac = a_smhv[4] / dv
        bc = b_smhv[4] / dv
        ac_err = a_smhv_err[4] / dv
        bc_err = b_smhv_err[4] / dv
        sst = 'SST1.0/1.0'
    if (ensci[j] == 1) & (sol[j] == 1.0) & (buff[j] == 0.1):
        betac = beta_1001
        ac = a_smhv[2] / dv
        bc = b_smhv[2] / dv
        ac_err = a_smhv_err[2] / dv
        bc_err = b_smhv_err[2] / dv
        sst = 'SST1.0/0.1'

    if (ensci[j] == 0) & (sol[j] == 1.0) & (buff[j] == 0.1):
        betac = beta_1001
        ac = a_smhv[5] / dv
        bc = b_smhv[5] / dv
        ac_err = a_smhv_err[5] / dv
        bc_err = b_smhv_err[5] / dv
        sst = 'SST1.0/0.1'

    if ensci[j] == 0:
        sondestr = 'SPC-6A'
        tfast = tfast_spc
    else:
        sondestr = 'EN-SCI'
        tfast = tfast_ecc

    out_name = f'{simlist[j]}_{teamlist[j]}'
    print(out_name)
    dft[j] = df[(df.Sim == simlist[j]) & (df.Team == teamlist[j])]
    dft[j] = dft[j].reset_index()
    
    dfm = dfi[(dfi.Sim == simlist[j]) & (dfi.Team == teamlist[j])]
    dfm = dfm.reset_index()

    eta = 1 + ac / 100 + bc / 100 * np.log10(dft[j]['Pair'])

    dft[j]['etac'] = eta

    dft[j]['PO3_dqa'] = 0.043085 * dft[j]['Tpump_cor'] * (dft[j]['IM'] - dft[j]['iB2']) / \
                        (1 * dft[j]['PFcor_kom'])
    dft[j]['PF_con'] = dft[j]['PFcor_kom']

    if (ensci[j] == 1) & (sol[j] == 1.0) & (buff[j] == 0.1):
        dft[j]['PO3_dqa'] = 0.043085 * dft[j]['Tpump_cor'] * (dft[j]['IM'] - dft[j]['iB2']) / \
                            (1 * dft[j]['PFcor_jma'])
        dft[j]['PF_con'] = dft[j]['PFcor_jma']

    dft[j]['PO3_trrm'] = 0.043085 * dft[j]['Tpump_cor'] * (dft[j]['Ifast_minib0_deconv_sm10']) / \
                         (1 * dft[j]['PFcor_jma'])

    dft[j]['a'] = ac
    dft[j]['b'] = bc
    dft[j]['a_err'] = ac_err
    dft[j]['b_err'] = bc_err

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

    # a
    dft[j]['d_im_bkg'] = (dft[j]['dI'] ** 2 + dft[j]['dib1'] ** 2) / ((dft[j]['IM'] - dft[j]['iB0']) ** 2)
    if year == '2017':
        dft[j]['d_im_bkg'] = (dft[j]['dI'] ** 2 + dft[j]['dib1'] ** 2) / ((dft[j]['IM'] - dft[j]['iB2']) ** 2)

    dft[j]['d_im_bkg_trm'] = (dft[j]['dI'] ** 2 + dft[j]['dib1'] ** 2 + dft[j]['dbeta'] ** 2) / \
                             ((dft[j]['IM'] - dft[j]['iB0'] - betac) ** 2)

    # b
    dft[j]['d_pfe_hum'] = ((dft[j]['unc_Cpf_kom'] / dft[j]['Cpf_kom']) ** 2) + \
                          ((dft[j]['dPhim'] * dft[j]['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2))
    dft[j]['d_pfe_hum_trrm'] = ((dft[j]['unc_Cpf_jma'] / dft[j]['Cpf_jma']) ** 2) + \
                               ((dft[j]['dPhim'] * dft[j]['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2))
    # c

    dft[j]['d_eta_c'] = (dft[j]['deta_c'] / dft[j]['eta_c']) ** 2
    dft[j]['d_eta_c_trm'] = ((ac_err ** 2 + (np.log10(dft[j]['Pair']) * bc_err) ** 2) / (
            1 + ac + bc * np.log10(dft[j]['Pair'])) ** 2) + opm_err ** 2
    # d
    dft[j]['d_eta_a'] = (dft[j]['deta_a'] / dft[j]['eta_a']) ** 2
    # e
    dft[j]['d_tpump'] = (dft[j]['dtpump'] / dft[j]['Tpump_cor']) ** 2

    dft[j]['tota_unc'] = \
        np.sqrt(dft[j]['d_im_bkg'] + dft[j]['d_pfe_hum'] + dft[j]['d_eta_c'] + dft[j]['d_eta_a'] + dft[j]['d_tpump'])

    dft[j]['tota_unc_trc'] = \
        np.sqrt(dft[j]['d_im_bkg_trm'] + dft[j]['d_pfe_hum_trrm'] + dft[j]['d_eta_c'] + dft[j]['d_eta_a'] + dft[j][
            'd_tpump'])

    dft[j]['tota_unc_trm'] = \
        np.sqrt(dft[j]['d_im_bkg_trm'] + dft[j]['d_pfe_hum_trrm'] + dft[j]['d_eta_c_trm'] + dft[j]['d_eta_a'] + dft[j][
            'd_tpump'])

    # put the unc in partial ozone values
    dff = pd.DataFrame()

    # dft[['Tsim', 'Pair','Tair','IM','I_conv_slow','Ifast_minib0_deconv_sm10']]
    dff['Sim_Time'] = dft[j]['Tsim']
    dff['Data_Index'] = 1
    if bool_pre:dff['Data_Index'] = -1
    dff['IM_Cell_Measured'] = dft[j]['IM']
    dff['IS_Cell_Slow'] = dft[j]['I_slow_conv']
    dff['IF_Cell_Fast_Smooth'] = dft[j]['Ifast_minib0']
    dff['IFDS_Cell_Fast_Deconv_Smooth'] = dft[j]['Ifast_minib0_deconv']
    if dff.at[0,'IFDS_Cell_Fast_Deconv_Smooth'] == 0:
        dff.at[0, 'IFDS_Cell_Fast_Deconv_Smooth'] = float('nan')
    if not bool_pre:
        dff['Pair_Pressure_ESC'] = dft[j]['Pair']
        try:
            dff['Tair_Pressure_ESC'] = dft[j]['Tair']
        except KeyError:
            dff['Tair_Pressure_ESC'] = np.nan
        try:dff['Pump_T_Ext'] = dft[j]['TPint']
        except KeyError: dff['Pump_T_Ext'] = np.nan
        dff['Pump_T_Int'] = dft[j]['TPext']
        dff['Pump_T_Cor'] = dft[j]['Tpump_cor']
        dff['Absorption_Eff'] = 1
        dff['Pump_Eff_Conv'] = dft[j]['PF_con']
        dff['Conversion_Eff_Conv'] = 1
        dff['Pump_Eff_TRCC'] = dft[j]['PFcor_jma']
        dff['Conversion_Eff_TRCC'] = dft[j]['etac']
        dff['PO3_Conv'] = dft[j]['PO3_dqa']
        dff['PO3_TRC'] = dft[j]['PO3_trrm']
        dff['PO3_TRCC'] = dft[j]['PO3_cal']
        dff['PO3_OPM'] = dft[j]['PO3_OPM']
        dff['Unc_PO3_Conv'] = dft[j]['tota_unc'] * dft[j]['PO3_dqa']
        dff['Unc_PO3_TRC'] = dft[j]['tota_unc_trc'] * dft[j]['PO3_trrm']
        dff['Unc_PO3_TRCC'] = dft[j]['tota_unc_trm'] * dft[j]['PO3_cal']
        dff['Unc_PO3_OPM'] = opm_err * dft[j]['PO3_OPM']
    dff = dff.set_index(['Sim_Time'])
    # dff['Conversion_Eff_TRCC'] = dff['Conversion_Eff_TRCC'].round(8)
    if not bool_pre:
        dff[['Data_Index', 'Pair_Pressure_ESC', 'Tair_Pressure_ESC', 'IM_Cell_Measured', 'IS_Cell_Slow',
             'IF_Cell_Fast_Smooth', 'IFDS_Cell_Fast_Deconv_Smooth', 'Pump_T_Ext', 'Pump_T_Int', 'Absorption_Eff',
             'Pump_Eff_Conv', 'Conversion_Eff_Conv', 'Pump_Eff_TRCC',  'PO3_Conv', 'PO3_TRC',
             'PO3_TRCC', 'PO3_OPM', 'Unc_PO3_Conv', 'Unc_PO3_TRC', 'Unc_PO3_TRCC', 'Unc_PO3_OPM']]=\
            dff[['Data_Index', 'Pair_Pressure_ESC', 'Tair_Pressure_ESC', 'IM_Cell_Measured', 'IS_Cell_Slow',
             'IF_Cell_Fast_Smooth', 'IFDS_Cell_Fast_Deconv_Smooth', 'Pump_T_Ext', 'Pump_T_Int', 'Absorption_Eff',
             'Pump_Eff_Conv', 'Conversion_Eff_Conv', 'Pump_Eff_TRCC',  'PO3_Conv', 'PO3_TRC',
             'PO3_TRCC', 'PO3_OPM', 'Unc_PO3_Conv', 'Unc_PO3_TRC', 'Unc_PO3_TRCC', 'Unc_PO3_OPM']].round(6)
    # print(list(dff))
    try: serial_nr = dff.at[0, 'SerialNr']
    except KeyError: serial_nr = np.nan
    if  bool_pre: decay_time = dft[j].at[0, 'decay_time']
    else: decay_time = np.nan

    dff = dff.round(6)

    if simlist[j] < 158: datet = '12-2009'
    else: datet = '08-2010'
    print(datet)

    dfm.at[0, 'total_massloss'] = dfm.at[0, 'Diff']



    variables = ['Date', 'Sim_Nr', 'Manifold_Port_Nr', 'Sonde_Type', 'Sonde_Code', 'SST',
                 'Pump_Flow_Rate_M', 'Pump_Flow_Rate_Gnd', 'SS_Weight_Losses_Ascent', 'IB0', 'IB1', 'IB2',
                 'Time_Response_Fast',
                 'Time_Response_Slow','Decay_Time', 'Stoichiometry_Slow']

    values = [datet, dft[j].at[0, 'Sim'], dft[j].at[0, 'Team'], sondestr,
              serial_nr, sst,
              round(dft[j].at[0, 'PFcor'] / 0.975, 2), round(dft[j].at[0, 'PFcor'], 2),
              round(dfm.at[0, 'total_massloss'], 2), dft[j].at[0, 'iB0'], dft[j].at[0, 'iB1'],
              dft[j].at[0, 'iB2'], tfast, tslow,decay_time, betac]

    # df.to_csv(header=False, index=False)
    dff = dff.replace(0, np.nan)


    with open(f'/home/poyraden/Analysis/JOSIEfiles/Paper/{year}/{out_name}{end}.csv', 'w') as fout:
        for q, a in zip(variables, values):
            fout.write('{0}={1} \n'.format(q, a))

        fout.write('\n')
        dff.to_csv(fout)

    # dff = dfi.append(dff)
    # dff.to_csv(f'/home/poyraden/Analysis/JOSIEfiles/Paper/2017/{out_name}.csv', index=False)