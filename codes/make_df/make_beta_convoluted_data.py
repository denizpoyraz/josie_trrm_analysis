import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from functions.convolution_functions import convolution, convolution_test, smooth_and_convolute, convolution_islow0
from homogenization_functions import pumpflow_efficiency,return_phipcor, VecInterpolate_log
from convolution_functions import convolution,convolution_islow0
from constant_variables import *

# now use this beta values * 0.1 for the deconvolution of the signal and make a DF

beta_spe = False
bool_9602 = True
bool_0910_decay = False
bool_2017 = False

year = '9602'
file_out = f'Josie{year}_deconv_2023paper.csv'
if bool_0910_decay:file_out = 'Josie0910_deconv_2023_decay_added_147-149.csv'
df = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_Data_2023paper.csv", low_memory=False)
if bool_2017:
    df = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_Data_2023paper_ib2.csv", low_memory=False)
    file_out = f'Josie{year}_deconv_2023paper_ib2.csv'

####            #######             ########

if bool_0910_decay:
    df = df[(df.Sim > 146) & (df.Sim < 150)]



# Josie9602_Data_updjma
df['TS'] = pd.to_datetime(df.Tsim, unit='s')

df['IMminusIB0'] = df['IM'] - df['iB0']

simlist = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Sim'])
teamlist = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Team'])
sol = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Sol'].tolist())
buff = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Buf'])
ensci = np.asarray(df.drop_duplicates(['Sim', 'Team'])['ENSCI'])

dft = {}
list_data = []

for j in range(len(simlist)):

    sondestr = ''
    adxstr = ''
    solstr = ''
    bufstr = ''

    af = 1
    beta = 0

    if (ensci[j] == 1) & (sol[j] == 0.5) & (buff[j] == 0.5): beta = beta_en0505
    if (ensci[j] == 1) & (sol[j] == 1.0) & (buff[j] == 1.0): beta = beta_en1010
    if (ensci[j] == 0) & (sol[j] == 0.5) & (buff[j] == 0.5): beta = beta_sp0505
    if (ensci[j] == 0) & (sol[j] == 1.0) & (buff[j] == 1.0): beta = beta_sp1010
    if (ensci[j] == 1) & (sol[j] == 1.0) & (buff[j] == 0.1): beta = beta_1001
    if (ensci[j] == 0) & (sol[j] == 1.0) & (buff[j] == 0.1): beta = beta_1001

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

    title = str(simlist[j]) + '_' + str(teamlist[j]) + '_' + adxstr + sondestr + solstr + '-' + bufstr + 'B'
    type = sondestr + ' ' + str(sol[j]) + '\% - ' + str(buff[j]) + 'B'
    sp = str(simlist[j]) + '-' + str(teamlist[j])
    ptitle = sp + ' ' + sondestr + ' ' + str(sol[j]) + '% - ' + str(buff[j]) + 'B'
    # print(title)

    dft[j] = df[(df.Sim == simlist[j]) & (df.Team == teamlist[j])]
    ### for data of every 12 seconds
    # dft[j] = dft[j].resample('12S', on='TS').mean().interpolate()
    # df = df.reset_index()
    dft[j] = dft[j].reset_index()
    if beta_spe:
        beta = dft[j].loc[0, 'beta']
    # print(title, beta)

    if bool_9602: dft[j]['TPext'] = dft[j]['TPint']

    Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv = \
        convolution(dft[j], 'IMminusIB0', 'IM', 'Tsim', beta, 1, sondestr)


    if bool_0910_decay:
        if (simlist[j] >= 147) & (simlist[j] < 150):

            decay_time = 1275
            iB1_var = (dft[j].at[0, 'iB1'] - dft[j].at[0, 'iB0']) * np.exp(-decay_time / (25 * 60))

            Islow_ib1_decay, Islow_conv_ib1_decay, Ifast_ib1_decay, Ifast_deconv_ib1_decay, Ifastminib0_ib1_decay, Ifastminib0_deconv_ib1_decay = \
                convolution_islow0(dft[j], iB1_var, 'IMminusIB0', 'IM', 'Tsim', beta, 1, sondestr)

            dft[j]['I_slow_ib1_decay'] = Islow_ib1_decay
            dft[j]['I_slow_conv_ib1_decay'] = Islow_conv_ib1_decay
            dft[j]['I_fast_ib1_decay'] = Ifast_ib1_decay
            dft[j]['Ifast_minib0_ib1_decay'] = Ifastminib0_ib1_decay
            dft[j]['Ifast_deconv_ib1_decay'] = Ifast_deconv_ib1_decay
            dft[j]['Ifast_minib0_deconv_ib1_decay'] = Ifastminib0_deconv_ib1_decay

    dft[j]['Islow'] = Islow
    dft[j]['Islow_conv'] = Islow_conv
    dft[j]['Ifast'] = Ifast
    dft[j]['Ifast_minib0'] = Ifastminib0
    dft[j]['Ifast_deconv'] = Ifast_deconv
    dft[j]['Ifast_minib0_deconv'] = Ifastminib0_deconv

    if bool_0910_decay:
        dft[j]['Ifast_minib0_deconv_sm10'] = dft[j]['Ifast_minib0_deconv_ib1_decay'].rolling(window=5, center=True).mean()
    if not bool_0910_decay:
        dft[j]['Ifast_minib0_deconv_sm10'] = dft[j]['Ifast_minib0_deconv'].rolling(window=5, center=True).mean()



    list_data.append(dft[j])

df_dc = pd.concat(list_data, ignore_index=True)

# df_dc.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_deconv_updjma.csv")
df_dc.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/" + file_out)
# df_dc.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie9602_deconv_updjma.csv")


# O3, O3_tot_opm = calculate_totO3OPM_var(dft[j], 'PO3_jma')
    # O3_deconv, O3_tot_opm = calculate_totO3OPM_var(dft[j], 'PO3_deconv_jma')
    # O3_deconv_sm, O3_tot_opm = calculate_totO3OPM_var(dft[j], 'PO3_deconv_jma_sm')
    #
    # dft[j]['tot_O3'] = O3
    # dft[j]['tot_O3_deconv'] = O3_deconv
    # dft[j]['O3_deconv_sm'] = O3_deconv_sm
    # dft[j]['tot_OPM'] = O3_tot_opm
