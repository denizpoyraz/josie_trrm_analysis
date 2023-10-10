import numpy as np
import pandas as pd
from convolution_functions import convolution,convolution_islow0, smooth_gaussian
from constant_variables import *

#update of the code 25/04 for HV smoothing
beta_spe = False
bool_9602 = False
bool_0910_decay = True
bool_2017 = False
bool_sm_hv = True
bool_2beta = False

year = '0910'
pre = ''
if bool_sm_hv: pre = '_sm_hv'
if bool_2beta: pre = '_2beta'
if bool_2beta and bool_sm_hv: pre = '_sm_hv_2beta'

file_out = f'Josie{year}_deconv_2023paper_HVbeta_noib0.csv'
# file_out = f'Josie{year}_deconv_2023paper{pre}.csv'

if bool_0910_decay:
    # file_out = f'Josie0910_deconv_2023_decay_added_147-149{pre}.csv'
    file_out = f'Josie0910_deconv_2023_decay_added_147-149{pre}_HVbeta_noib0.csv'

df = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_Data_2023paper.csv", low_memory=False)
if bool_2017:
    df = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_Data_2023paper_ib2.csv", low_memory=False)
    # file_out = f'Josie{year}_deconv_2023paper_ib2{pre}.csv'
    file_out = f'Josie{year}_deconv_2023paper_HVbeta_noib0.csv'


####            #######             ########

if bool_0910_decay:
    df = df[(df.Sim > 146) & (df.Sim < 150)]


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

    if bool_2beta: beta = 2 *beta

    if ensci[j] == 0:
        sondestr = 'SPC'
        tfast = tfast_spc
    else:
        sondestr = 'ENSCI'
        tfast = tfast_ecc


    sigma = 0.2 * tfast
    timew = 3 * sigma

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

    dft[j] = dft[j].reset_index()
    if beta_spe:
        beta = dft[j].loc[0, 'beta']

    if bool_9602: dft[j]['TPext'] = dft[j]['TPint']
    #update everthing to HV smoothed IM

    dft[j]['IM_gsm'] = smooth_gaussian(dft[j], 'Tsim', 'IM', timew, sigma)
    dft[j]['IMminusIB0_gsm'] = smooth_gaussian(dft[j], 'Tsim', 'IMminusIB0', timew, sigma)

    Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv = \
        convolution(dft[j], 'IMminusIB0', 'IM', 'Tsim', beta, 1, sondestr)
    if bool_sm_hv:
        Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv = \
            convolution(dft[j], 'IMminusIB0_gsm', 'IM_gsm', 'Tsim', beta, 1, sondestr)


    if bool_0910_decay:
        if (simlist[j] >= 147) & (simlist[j] < 150):

            decay_time = 1275
            iB1_var = (dft[j].at[0, 'iB1'] - dft[j].at[0, 'iB0']) * np.exp(-decay_time / (25 * 60))

            Islow_ib1_decay, Islow_conv_ib1_decay, Ifast_ib1_decay, Ifast_deconv_ib1_decay, Ifastminib0_ib1_decay, Ifastminib0_deconv_ib1_decay = \
                convolution_islow0(dft[j], iB1_var, 'IMminusIB0', 'IM', 'Tsim', beta, 1, sondestr)
            if bool_sm_hv:
                Islow_ib1_decay, Islow_conv_ib1_decay, Ifast_ib1_decay, Ifast_deconv_ib1_decay, Ifastminib0_ib1_decay, Ifastminib0_deconv_ib1_decay = \
                convolution_islow0(dft[j], iB1_var, 'IMminusIB0_gsm', 'IM_gsm', 'Tsim', beta, 1, sondestr)

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
        if bool_sm_hv:
            dft[j]['Ifast_minib0_deconv_sm10'] = dft[j]['Ifast_minib0_deconv_ib1_decay']
    if not bool_0910_decay:
        dft[j]['Ifast_minib0_deconv_sm10'] = dft[j]['Ifast_minib0_deconv'].rolling(window=5, center=True).mean()
        if bool_sm_hv:
            dft[j]['Ifast_minib0_deconv_sm10'] = dft[j]['Ifast_minib0_deconv']


    list_data.append(dft[j])

df_dc = pd.concat(list_data, ignore_index=True)

df_dc.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/" + file_out)
