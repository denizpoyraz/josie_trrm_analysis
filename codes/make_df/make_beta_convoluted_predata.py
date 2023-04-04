import pandas as pd
from convolution_functions import convolution, convolution_islow0
from constant_variables import *

## updated version of Beta_Convoluted_final to consider I slow = I prep(Tprep_end)

def mederr(med):
    err = (np.nanquantile(med, 0.75) - np.nanquantile(med, 0.25)) / (2 * 0.6745)
    return err


def filter(df):
    filtEN = df.ENSCI == 1
    filtSP = df.ENSCI == 0

    filtS10 = df.Sol == 1
    filtS05 = df.Sol == 0.5

    filtB10 = df.Buf == 1.0
    filtB05 = df.Buf == 0.5

    filterEN0505 = (filtEN & filtS05 & filtB05)
    filterEN1010 = (filtEN & filtS10 & filtB10)

    profEN0505 = df.loc[filterEN0505]
    profEN1010 = df.loc[filterEN1010]
    profEN0505_nodup = profEN0505.drop_duplicates(['Sim', 'Team'])
    profEN1010_nodup = profEN1010.drop_duplicates(['Sim', 'Team'])

    ###
    filterSP1010 = (filtSP & filtS10 & filtB10)
    filterSP0505 = (filtSP & filtS05 & filtB05)

    profSP1010 = df.loc[filterSP1010]
    profSP0505 = df.loc[filterSP0505]
    profSP1010_nodup = profSP1010.drop_duplicates(['Sim', 'Team'])
    profSP0505_nodup = profSP0505.drop_duplicates(['Sim', 'Team'])

    sim_en0505 = profEN0505_nodup.Sim.tolist()
    team_en0505 = profEN0505_nodup.Team.tolist()

    sim_en1010 = profEN1010_nodup.Sim.tolist()
    team_en1010 = profEN1010_nodup.Team.tolist()

    sim_sp0505 = profSP0505_nodup.Sim.tolist()
    team_sp0505 = profSP0505_nodup.Team.tolist()

    sim_sp1010 = profSP1010_nodup.Sim.tolist()
    team_sp1010 = profSP1010_nodup.Team.tolist()

    sim = [sim_en0505, sim_en1010, sim_sp0505, sim_sp1010]
    team = [team_en0505, team_en1010, team_sp0505, team_sp1010]

    return sim, team


# now use this beta values for the deconvolution of the signal and make a DF

# df = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_preparationadded_updjma.csv", low_memory=False)
df = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_preparationadded_2023paper.csv", low_memory=False)

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

# teamlist = [1,2,3,4]

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
    dfto[j] = df[(df.Sim == simlist[j]) & (df.Team == teamlist[j]) & (df.TimeTag == 'Sim')]

    ### for data of every 12 seconds
    # dft[j] = dft[j].resample('12S', on='TS').mean().interpolate()
    # df = df.reset_index()
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


    Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv = convolution(dft[j], 'IMminusIB0', 'IM',
                                                                                          'Tsim', beta, 1, sondestr)
    Islowo, Islow_convo, Ifasto, Ifast_deconvo, Ifastminib0o, Ifastminib0_deconvo = convolution(dfto[j], 'IMminusIB0',
                                                                                                'IM', 'Tsim_original',
                                                                                                beta, 1, sondestr)

    Islow_ib1_decay, Islow_conv_ib1_decay, Ifast_ib1_decay, Ifast_deconv_ib1_decay, Ifastminib0_ib1_decay, Ifastminib0_deconv_ib1_decay = \
        convolution_islow0(dfto[j], iB1_var, 'IMminusIB0', 'IM', 'Tsim_original', beta, 1, sondestr)

    Islow_ib1_test, Islow_conv_ib1_test, Ifast_ib1_test, Ifast_deconv_ib1_test, Ifastminib0_ib1_test, Ifastminib0_deconv_ib1_test = \
        convolution_islow0(dfto[j], test_var, 'IMminusIB0', 'IM', 'Tsim_original', beta, 1, sondestr)

    Islow_ib1_decay_all, Islow_conv_ib1_decay_all, Ifast_ib1_decay_all, Ifast_deconv_ib1_decay_all, Ifastminib0_ib1_decay_all, Ifastminib0_deconv_ib1_decay_all = \
        convolution_islow0(dft[j], iB1_var, 'IMminusIB0', 'IM', 'Tsim', beta, 1, sondestr)

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

df_dc.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Deconv_preparationadded_all_2023paper.csv")
df_o.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Deconv_preparationadded_simulation_2023paper.csv")

