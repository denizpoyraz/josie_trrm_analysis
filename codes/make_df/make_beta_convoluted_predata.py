import pandas as pd
from convolution_functions import convolution, convolution_islow0, smooth_gaussian
from constant_variables import *


bool_sm_hv = True

df = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_preparationadded_2023paper.csv", low_memory=False)

pre = ''
if bool_sm_hv: pre = '_sm_hv'

df['PF'] = df['PFcor']
df['TS'] = pd.to_datetime(df.Tsim, unit='s')


df = df.reset_index(drop=True)

df = df.drop(df[(df.Sim == 158) & (df.Team == 2)].index)
df = df.drop(df[(df.Sim == 160) & (df.Team == 4)].index)

simlist = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Sim'])
teamlist = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Team'])
# adx = np.asarray(df.drop_duplicates(['Sim', 'Team'])['ADX'])
sol = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Sol'].tolist())
buff = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Buf'])
ensci = np.asarray(df.drop_duplicates(['Sim', 'Team'])['ENSCI'])


print(simlist, teamlist)

dft = {}
list_data = []

dfto = {}
list_data_original = []
##now try to use a different dataset: data every 12 seconds

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
        tfast = tfast_spc
    else:
        sondestr = 'ENSCI'
        tfast = tfast_ecc

    if sol[j] == 2.0: solstr = '2p0'
    if sol[j] == 1.0: solstr = '1p0'
    if sol[j] == 0.5: solstr = '0p5'

    if buff[j] == 0.1: bufstr = '0p1'
    if buff[j] == 0.5: bufstr = '0p5'
    if buff[j] == 1.0: bufstr = '1p0'
    
    sigma = 0.2 * tfast
    timew = 3 * sigma

    title = str(simlist[j]) + '_' + str(teamlist[j]) + '_' + adxstr + sondestr + solstr + '-' + bufstr + 'B'
    type = sondestr + ' ' + str(sol[j]) + '\% - ' + str(buff[j]) + 'B'
    sp = str(simlist[j]) + '-' + str(teamlist[j])
    ptitle = sp + ' ' + sondestr + ' ' + str(sol[j]) + '% - ' + str(buff[j]) + 'B'
    # print(title)

    dft[j] = df[(df.Sim == simlist[j]) & (df.Team == teamlist[j])]
    dfto[j] = df[(df.Sim == simlist[j]) & (df.Team == teamlist[j]) & (df.TimeTag == 'Sim')]

    dft[j] = dft[j].reset_index(drop=True)
    dfto[j] = dfto[j].reset_index(drop=True)


    dft[j]['IMminusIB0'] = dft[j]['IM'] - dft[j]['iB0']
    dfto[j]['IMminusIB0'] = dfto[j]['IM'] - dfto[j]['iB0']


    dft[j]['Tsim_min'] = dft[j].Tsim / 60
    iprep_max = dft[j][dft[j].Tsim_min < 40].IM.max()
    iprep_max_index = dft[j][dft[j].IM == iprep_max].index.values
    time_iprep_max = dft[j][dft[j].IM == iprep_max]['Tsim_min'].tolist()[0]
    time_ib1 = time_iprep_max + 10
    i_at_timeib1 = np.median(
        dft[j][(dft[j].Tsim_min < time_ib1 + 0.1) & (dft[j].Tsim_min > time_ib1 - 0.1)]['IM'].tolist())
    simbegin_time = dft[j][dft[j].Tsim_original == 10]['Tsim'].tolist()[0]
    #
    # imax begin flight
    if (simlist[j] != 140):
        imax_bf = dfto[j][(dfto[j].Tsim_original < 900) & (dfto[j].Tsim_original > 500) & (dfto[j].IM > 0.1) & (
                dfto[j].IM < 0.4)].IM.tolist()
        imax_bf = np.min(imax_bf)
        imax_bf_time = dfto[j][(dfto[j].IM == imax_bf) & (dfto[j].Tsim_original < 900) & (dfto[j].Tsim_original > 500)][
            'Tsim_original'].tolist()[0]
        bf_time = imax_bf_time - 6

    if (simlist[j] == 140):
        imax_bf = dfto[j][(dfto[j].Tsim_original < 1100) & (dfto[j].Tsim_original > 975) & (dfto[j].IM > 0.1) & (
                dfto[j].IM < 0.4)].IM.tolist()
        imax_bf = np.min(imax_bf)
        imax_bf_time = \
            dfto[j][(dfto[j].IM == imax_bf) & (dfto[j].Tsim_original < 1100) & (dfto[j].Tsim_original > 975)][
                'Tsim_original'].tolist()[0]
        bf_time = imax_bf_time - 6

    decay_time = simbegin_time - time_ib1 * 60

    dft[j]['decay_time'] = decay_time

    iB1_var = (dft[j].at[0, 'iB1'] - dft[j].at[0, 'iB0']) * np.exp(-decay_time / (25 * 60))
    test_var = (dft[j].at[0, 'iB1'] - dft[j].at[0, 'iB0'])

    dft[j]['IM_gsm'] = smooth_gaussian(dft[j], 'Tsim', 'IM', timew, sigma)
    dft[j]['IMminusIB0_gsm'] = smooth_gaussian(dft[j], 'Tsim', 'IMminusIB0', timew, sigma)
    
    dfto[j]['IM_gsm'] = smooth_gaussian(dfto[j], 'Tsim_original', 'IM', timew, sigma)
    dfto[j]['IMminusIB0_gsm'] = smooth_gaussian(dfto[j], 'Tsim_original', 'IMminusIB0', timew, sigma)

    Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv = convolution(dft[j], 'IMminusIB0', 'IM',
                                                                                          'Tsim', beta, 1, sondestr)
    Islowo, Islow_convo, Ifasto, Ifast_deconvo, Ifastminib0o, Ifastminib0_deconvo = convolution(dfto[j], 'IMminusIB0',
                                                                                                'IM', 'Tsim_original',
                                                                                                beta, 1, sondestr)
    if bool_sm_hv:
        Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv = \
            convolution(dft[j], 'IMminusIB0_gsm', 'IM_gsm','Tsim', beta, 1, sondestr)
        Islowo, Islow_convo, Ifasto, Ifast_deconvo, Ifastminib0o, Ifastminib0_deconvo = \
            convolution(dfto[j], 'IMminusIB0_gsm','IM_gsm', 'Tsim_original', beta, 1, sondestr)
        

    Islow_ib1_decay, Islow_conv_ib1_decay, Ifast_ib1_decay, Ifast_deconv_ib1_decay, Ifastminib0_ib1_decay, Ifastminib0_deconv_ib1_decay = \
        convolution_islow0(dfto[j], iB1_var, 'IMminusIB0', 'IM', 'Tsim_original', beta, 1, sondestr)
    Islow_ib1_test, Islow_conv_ib1_test, Ifast_ib1_test, Ifast_deconv_ib1_test, Ifastminib0_ib1_test, Ifastminib0_deconv_ib1_test = \
        convolution_islow0(dfto[j], test_var, 'IMminusIB0', 'IM', 'Tsim_original', beta, 1, sondestr)
    Islow_ib1_decay_all, Islow_conv_ib1_decay_all, Ifast_ib1_decay_all, Ifast_deconv_ib1_decay_all, Ifastminib0_ib1_decay_all, Ifastminib0_deconv_ib1_decay_all = \
        convolution_islow0(dft[j], iB1_var, 'IMminusIB0', 'IM', 'Tsim', beta, 1, sondestr)

    if bool_sm_hv:
        Islow_ib1_decay, Islow_conv_ib1_decay, Ifast_ib1_decay, Ifast_deconv_ib1_decay, Ifastminib0_ib1_decay, Ifastminib0_deconv_ib1_decay = \
            convolution_islow0(dfto[j], iB1_var, 'IMminusIB0_gsm', 'IM_gsm', 'Tsim_original', beta, 1, sondestr)
        Islow_ib1_test, Islow_conv_ib1_test, Ifast_ib1_test, Ifast_deconv_ib1_test, Ifastminib0_ib1_test, Ifastminib0_deconv_ib1_test = \
            convolution_islow0(dfto[j], test_var, 'IMminusIB0_gsm', 'IM_gsm', 'Tsim_original', beta, 1, sondestr)
        Islow_ib1_decay_all, Islow_conv_ib1_decay_all, Ifast_ib1_decay_all, Ifast_deconv_ib1_decay_all, Ifastminib0_ib1_decay_all, Ifastminib0_deconv_ib1_decay_all = \
            convolution_islow0(dft[j], iB1_var, 'IMminusIB0_gsm', 'IM_gsm', 'Tsim', beta, 1, sondestr)

    dft[j]['i_at_timeib1'] = i_at_timeib1
    dft[j]['time_ib1'] = time_ib1
    dft[j]['decay_time'] = decay_time
    dft[j]['start_flight'] = bf_time

    dfto[j]['i_at_timeib1'] = i_at_timeib1
    dfto[j]['time_ib1'] = time_ib1
    dfto[j]['decay_time'] = decay_time
    dfto[j]['start_flight'] = bf_time

    dft[j]['I_slow'] = Islow
    dft[j]['I_slow_conv'] = Islow_conv
    dft[j]['I_fast'] = Ifast
    dft[j]['Ifast_minib0'] = Ifastminib0
    dft[j]['Ifast_deconv'] = Ifast_deconv
    dft[j]['Ifast_minib0_deconv'] = Ifastminib0_deconv
    dft[j]['Ifast_minib0_deconv_sm10'] = dft[j]['Ifast_minib0_deconv'].rolling(window=5, center=True).mean()


    dfto[j]['I_slowo'] = Islowo
    dfto[j]['I_slow_convo'] = Islow_convo
    dfto[j]['I_fasto'] = Ifasto
    dfto[j]['Ifast_minib0o'] = Ifastminib0o
    dfto[j]['Ifast_deconvo'] = Ifast_deconvo
    dfto[j]['Ifast_minib0_deconvo'] = Ifastminib0_deconvo
    dfto[j]['Ifast_minib0_deconv_sm10'] = dfto[j]['Ifast_minib0_deconvo'].rolling(window=5, center=True).mean()

    dfto[j]['I_slow_ib1_decay'] = Islow_ib1_decay
    dfto[j]['I_slow_conv_ib1_decay'] = Islow_conv_ib1_decay
    dfto[j]['I_fast_ib1_decay'] = Ifast_ib1_decay
    dfto[j]['Ifast_minib0_ib1_decay'] = Ifastminib0_ib1_decay
    dfto[j]['Ifast_deconv_ib1_decay'] = Ifast_deconv_ib1_decay
    dfto[j]['Ifast_minib0_deconv_ib1_decay'] = Ifastminib0_deconv_ib1_decay
    dfto[j]['Ifast_minib0_deconv_ib1_decay_sm10'] = dfto[j]['Ifast_minib0_deconv_ib1_decay'].rolling(window=5, center=True).mean()


    dfto[j]['I_slow_ib1_test'] = Islow_ib1_test
    dfto[j]['I_slow_conv_ib1_test'] = Islow_conv_ib1_test
    dfto[j]['I_fast_ib1_test'] = Ifast_ib1_test
    dfto[j]['Ifast_minib0_ib1_test'] = Ifastminib0_ib1_test
    dfto[j]['Ifast_deconv_ib1_test'] = Ifast_deconv_ib1_test
    dfto[j]['Ifast_minib0_deconv_ib1_test'] = Ifastminib0_deconv_ib1_test

    dft[j]['I_slow_conv_ib1_decay_all'] = Islow_conv_ib1_decay_all
    dft[j]['I_fast_ib1_decay_all'] = Ifast_ib1_decay_all
    dft[j]['Ifast_minib0_ib1_decay_all'] = Ifastminib0_ib1_decay_all
    dft[j]['Ifast_deconv_ib1_decay_all'] = Ifast_deconv_ib1_decay_all
    dft[j]['Ifast_minib0_deconv_ib1_decay_all'] = Ifastminib0_deconv_ib1_decay_all

    
    

    list_data.append(dft[j])
    list_data_original.append(dfto[j])

df_dc = pd.concat(list_data, ignore_index=True)
df_o = pd.concat(list_data_original, ignore_index=True)

df_dc.to_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Deconv_preparationadded_all_2023paper{pre}.csv")
df_o.to_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Deconv_preparationadded_simulation_2023paper{pre}.csv")

