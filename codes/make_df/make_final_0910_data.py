import pandas as pd
import numpy as np

##merge 0910 preparation data with the simulation data
# there were some simulations of 0910 that did not have preparation data,
# those simulations are reprocessed in code make_beta_convoluted_data with boolean 0910_decay = True
# this codes merges the output of make_beta_convoluted_preparation_data and make_beta_convoluted_data
# the final df to use for Josie 0910 dataset

df1 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/"
                  "Josie0910_Deconv_preparationadded_simulation_2023paper_sm_hv.csv",low_memory=False)
df1 = df1[df1.ADX ==0]

df1t = df1.drop_duplicates(['Team','Sim'])
decay_time = df1t.decay_time.median()

df1 = df1[df1.Tsim_original >= 0]
df1['Tsim'] = df1['Tsim_original'].copy()
df1['Ifast_minib0_deconv'] = df1['Ifast_minib0_deconvo'].copy()

df2 = pd.read_csv('/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_deconv_2023_decay_added_147-149_sm_hv.csv')
df2 = df2[df2.ADX ==0]

print('list df1', list(df1))
print('list df2', list(df2))

df1['Ifast_minib0'] = df1['Ifast_minib0o']
df1['Islow_conv'] = df1['I_slow_convo']


df1c = df1[['Sim','Team', 'iB0', 'iB1', 'iB2', 'ENSCI', 'Sol', 'ADX','Buf','PO3','PO3_OPM', 'Tsim','Pair',
            'IM', 'I_OPM_kom', 'I_OPM_jma','PFcor','TPext','Tpump_cor','unc_Tpump_cor','unc_Tpump', 'Tpump_cor',
            'unc_Tpump_cor', 'deltat', 'unc_deltat','deltat_ppi', 'unc_deltat_ppi', 'Cpf_kom', 'unc_Cpf_kom',
            'Cpf', 'unc_Cpf','Cpf_jma', 'unc_Cpf_jma',
            'PFcor_kom', 'PFcor_jma', 'PO3_calc', 'PO3_dqa','I_slow_conv_ib1_decay','Ifast_minib0_deconv',
            'Ifast_minib0_deconv_ib1_decay','Ifast_minib0', 'Ifast_minib0_ib1_decay', 'Islow_conv']].copy()


df2c = df2[['Sim','Team', 'iB0', 'iB1', 'iB2', 'ENSCI', 'Sol', 'ADX','Buf','PO3','PO3_OPM', 'Tsim','Pair',
            'IM', 'I_OPM_kom', 'I_OPM_jma','PFcor','TPext','Tpump_cor','unc_Tpump_cor','unc_Tpump', 'Tpump_cor',
            'unc_Tpump_cor', 'deltat', 'unc_deltat','deltat_ppi', 'unc_deltat_ppi', 'Cpf_kom', 'unc_Cpf_kom',
            'Cpf', 'unc_Cpf','Cpf_jma', 'unc_Cpf_jma',
            'PFcor_kom', 'PFcor_jma', 'PO3_calc', 'PO3_dqa','I_slow_conv_ib1_decay',
            'Ifast_minib0_deconv', 'Ifast_minib0_deconv_ib1_decay','Ifast_minib0', 'Ifast_minib0_ib1_decay','Islow_conv']].copy()

df_0910_final = pd.concat([df1c, df2c], ignore_index=True)
df_0910_final.to_csv('/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_deconv_2023_unitedpaper_sm_hv.csv')