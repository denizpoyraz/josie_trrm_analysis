import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from constant_variables import *
from convolution_functions import smooth_parabolic, smooth_gaussian, convolution_df


# df = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_deconv_2023paper.csv")
df = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_deconv_2023_unitedpaper.csv")
df = df[df.ADX == 0]

# df = df[(df.Buf == 1.0) & (df.ENSCI == 1)]
df = df[(df.Sim == 136) & (df.Team ==3)]
df['iB0'] = 0.03
# df = df[(df.Sim == 146)]
#
# df = df[(df.Sim == 141) | (df.Sim == 147) | (df.Sim == 158) | (df.Sim == 159) | (df.Sim == 167) | (df.Sim == 163)
# | (df.Sim == 160) | (df.Sim == 165)]
# df1t = df1[(df1.Sim == 136) & (df1.Team ==3)]
# df2t = df2[(df2.Sim == 136) & (df2.Team ==3)]
# df2t = df2t[df2t.Tsim_original >= 0]
# df3t = df3[(df3.Sim == 136) & (df3.Team ==3)]



# df['i_fast_par_sm20'] = smooth_parabolic(df, 'Ifast_minib0_deconv', df.Tsim, 10)
df['Ifast_minib0_deconv_sm20'] = df['Ifast_minib0_deconv'].rolling(window=9, center=True).mean()
df['Ifast_minib0_deconv_sm10'] = df['Ifast_minib0_deconv'].rolling(window=5, center=True).mean()
df['Ifast_minib0_deconv_sm5'] = df['Ifast_minib0_deconv'].rolling(window=3, center=True).mean()

simlist = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Sim'])
teamlist = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Team'])
sol = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Sol'].tolist())
buff = np.asarray(df.drop_duplicates(['Sim', 'Team'])['Buf'])
ensci = np.asarray(df.drop_duplicates(['Sim', 'Team'])['ENSCI'])

dft = {}

prefix = ''

cbl = ['#e41a1c', '#a65628','#dede00', '#4daf4a', '#377eb8', '#984ea3']
#red, brown, yellow,
print(simlist, teamlist)

for j in range(len(simlist)):
    sigma = 0
    if ensci[j] == 0:
        sondestr = 'SPC'
        print('SPC')
        sigma = 0.2 * tfast_spc
        timew = 3 * sigma
    else:
        print('ENSCI')
        sondestr = 'ENSCI'
        sigma = 0.2 * tfast_ecc
        timew = 3 * sigma

    if sol[j] == 2.0: solstr = '2p0'
    if sol[j] == 1.0: solstr = '1p0'
    if sol[j] == 0.5: solstr = '0p5'

    if buff[j] == 0.1: bufstr = '0p1'
    if buff[j] == 0.5: bufstr = '0p5'
    if buff[j] == 1.0: bufstr = '1p0'

    if (ensci[j] == 1) & (sol[j] == 0.5) & (buff[j] == 0.5):
        beta = beta_en0505
        sst = 'SST 0.5'
    if (ensci[j] == 1) & (sol[j] == 1.0) & (buff[j] == 1.0):
        beta = beta_en1010
        sst = 'SST 1.0'
    if (ensci[j] == 0) & (sol[j] == 0.5) & (buff[j] == 0.5):
        beta = beta_sp0505
        sst = 'SST 0.5'
    if (ensci[j] == 0) & (sol[j] == 1.0) & (buff[j] == 1.0):
        beta = beta_sp1010
        sst = 'SST 1.0'

    # if ensci[j] == 1:continue
    # if sol[j] != 1.0: continue

    title = prefix + str(simlist[j]) + '_' + str(teamlist[j])
    type = sondestr + ' ' + str(sol[j]) + '\% - ' + str(buff[j]) + 'B'
    sp = str(simlist[j]) + '-' + str(teamlist[j])
    # ptitle =   sp + ' ' + sondestr + ' ' + str(sol[j]) + '% - ' + str(buff[j]) + 'B'
    ptitle =  f'{sp} {sondestr} {sst} '

    print(title)

    dft[j] = df[(df.Sim == simlist[j]) & (df.Team == teamlist[j])]
    # for data that has pre-simulation
    # dft[j] = df[(df.Sim == simlist[j]) & (df.Team == teamlist[j]) & (df.TimeTag == 'Sim')]
    dft[j] = dft[j].reset_index()
    size = len(dft[j])
    dft[j]['i_fast_par_sm10'] = smooth_parabolic(dft[j], 'Ifast_minib0_deconv', dft[j].Tsim, 5)
    dft[j]['IminusiB0'] = dft[j]['IM'] - dft[j]['iB0']

    # I_s_ghv = smooth_gaussian(dft[j], 'Tsim','IM', timew, sigma)
    # Imib0_s_ghv = smooth_gaussian(dft[j],'Tsim', 'IminusiB0', timew, sigma)

    dft[j]['I_gsm'] = smooth_gaussian(dft[j], 'Tsim','IM', timew, sigma)
    dft[j]['IminiB0_gsm'] = smooth_gaussian(dft[j],'Tsim', 'IminusiB0', timew, sigma)
    dft[j]['Ifast_minib0_deconv_gs'] = convolution_df(dft[j], 'IminiB0_gsm', 'I_gsm', 'Tsim', beta, 'ENSCI')

    size = len(dft[j])
    Ums_i = [0] * size
    Ua_i = [0] * size

    Ums_i[0] = dft[j].at[0, 'IM']

    for i in range(size - 1):
        # Ua_i = dft[j].at[i+1, 'IMminusiB0']
        Ua_i[i] = dft[j].at[i, 'I_OPM_jma']
        Ua_i[i + 1] = dft[j].at[i + 1, 'I_OPM_jma']
        t1 = dft[j].at[i + 1, 'Tsim']
        t2 = dft[j].at[i, 'Tsim']
        Xs = np.exp(-(t1 - t2) / tslow)
        Ums_i[i + 1] = Ua_i[i + 1] - (Ua_i[i + 1] - Ums_i[i]) * Xs
    # fi = dft[j].first_valid_index()
    # li = dft[j].last_valid_index()
    dft[j].loc[:, 'I_conv_slow_jma'] = Ums_i
    size_label = 22
    size_title = 24
    size_text = 18
    dft[j]['Tsim'] = dft[j]['Tsim']/60

    plt.figure(figsize=(12, 8))
    plt.ylabel(r'Current [$\mu$A])', fontsize=size_label)
    # plt.xlabel(r'Simulation time $[sec.]$', fontsize = size_label)
    plt.xlabel(r'Simulation time $[min.]$', fontsize = size_label)

    plt.yticks(fontsize=size_label)
    plt.xticks(fontsize=size_label)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())

    plt.gca().tick_params(which='major', width=2)
    plt.gca().tick_params(which='minor', width=2)
    plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
    plt.gca().xaxis.set_tick_params(length=5, which='minor')
    plt.gca().xaxis.set_tick_params(length=10, which='major')
    plt.gca().yaxis.set_tick_params(length=5, which='minor')
    plt.gca().yaxis.set_tick_params(length=10, which='major')

    #
    # pre = 'fig2_5_'
    # plt.plot(dft[j].Tsim, dft[j].IM - dft[j].at[0,'iB0'], label = r'I$_M$ - I$_{B0}$', color =cbl[4], linewidth=0.75)
    # plt.plot(dft[j].Tsim, dft[j].I_OPM_jma, label=r'I$_{OPM}$', color = cbl[0],linewidth=0.75)
    # plt.plot(dft[j].Tsim, dft[j].Ifast_minib0_ib1_decay, label=r'I$_{F}$', color = cbl[3], linewidth=0.75) #green
    # plt.plot(dft[j].Tsim, dft[j].I_conv_slow_jma, label=r'I$_{OPM,C}$', color = cbl[2], linewidth=2)
    # plt.plot(dft[j].Tsim, beta * dft[j].I_conv_slow_jma, label=r'S$_{S}\times$I$_{OPM,C}$', color = cbl[1], linewidth=1.5)
    # #
    # plt.text((2165-150)/60, 1, 'RT1', weight='bold', fontsize=size_text, bbox={'facecolor': 'white', 'alpha': 0.6})
    # plt.text((4100-150)/60, 4.5, 'RT2', weight='bold',fontsize=size_text, bbox={'facecolor': 'white', 'alpha': 0.6})
    # plt.text((6200-150)/60, 4.5, 'RT3', weight='bold', fontsize=size_text,bbox={'facecolor': 'white', 'alpha': 0.6})
    # plt.text((8160-150)/60, 1.4, 'RT4', weight='bold', fontsize=size_text,bbox={'facecolor': 'white', 'alpha': 0.6})

    #fig 5
    # pre = 'fig_5_'
    # plt.plot(dft[j].Tsim, dft[j].IM - dft[j].at[0,'iB0'], label = r'I$_M$ - I$_{B0}$', color =cbl[4]) #blue
    # plt.plot(dft[j].Tsim, dft[j].I_OPM_jma, label=r'I$_{OPM}$', color = cbl[0]) #red
    # plt.plot(dft[j].Tsim, dft[j].Ifast_minib0_ib1_decay, label=r'I$_{F}$', color = cbl[3]) #green
    # plt.plot(dft[j].Tsim, dft[j].Islow_conv, label=r'I$_{S}$', color = cbl[1]) #brown
    #
    # plt.text(2165-150, 1, 'RT1', weight='bold', fontsize=size_text, bbox={'facecolor': 'white', 'alpha': 0.6})
    # plt.text(4100-150, 4.5, 'RT2', weight='bold',fontsize=size_text, bbox={'facecolor': 'white', 'alpha': 0.6})
    # plt.text(6200-150, 4.5, 'RT3', weight='bold', fontsize=size_text,bbox={'facecolor': 'white', 'alpha': 0.6})
    # plt.text(8160-150, 1.4, 'RT4', weight='bold', fontsize=size_text,bbox={'facecolor': 'white', 'alpha': 0.6})

    # plt.ylim(0.001, 10)
    # plt.xlim(0, 8500)
    # plt.xlim(0, 145)
    # plt.yscale('log')


    # #fig 6
    pre = 'fig_5_'
    #
    plt.plot(dft[j].Tsim, dft[j].IM - dft[j].at[0,'iB0'], label = r'I$_M$ - I$_{B0}$', color =cbl[4]) #blue
    plt.plot(dft[j].Tsim, dft[j].I_OPM_jma, label=r'I$_{OPM}$', color = cbl[0]) #red
    plt.plot(dft[j].Tsim, dft[j].Ifast_minib0_deconv_ib1_decay, label=r'I$_{F,D}$', color = cbl[2]) #yellow
    # # plt.plot(dft[j].Tsim, dft[j].Ifast_minib0_ib1_decay, label=r'I$_{F}$', color = cbl[2]) #yellow
    #
    # #
    # gsm_label = f'sigma=0.2'
    #
    plt.plot(dft[j].Tsim, dft[j].Ifast_minib0_deconv_sm10, label=r'I$_{F,D}$ (running averages 10 sec.)', color = cbl[1]) #red
    plt.plot(dft[j].Tsim, dft[j].Ifast_minib0_deconv_sm20, label=r'I$_{F,D}$ (running averages 20 sec.)', color = cbl[5]) #red
    plt.plot(dft[j].Tsim, dft[j].Ifast_minib0_deconv_gs,
             label=r'I$_{F,D}$ (gaussian smoothing $\sigma=0.2\cdot\tau_f$, window=3$\cdot\sigma$)',
             color=cbl[3])  # red
    plt.ylim(-0.09, 0.9)
    plt.xlim(1900/60, 2600/60)
    # cbl = ['#e41a1c', '#a65628', '#dede00', '#4daf4a', '#377eb8', '#984ea3']
    # red, brown, yellow, green, blue, purple

    # df['i_fast_par_sm10'] = smooth_parabolic(df, 'Ifast_minib0_deconv', df.Tsim, 5)
    # df['i_fast_par_sm20'] = smooth_parabolic(df, 'Ifast_minib0_deconv', df.Tsim, 10)
    # df['Ifast_minib0_deconv_sm20'] = df['Ifast_minib0_deconv'].rolling(window=9, center=True).mean()
    # df['Ifast_minib0_deconv_sm10'] = df['Ifast_minib0_deconv'].rolling(window=5, center=True).mean()
    # df['Ifast_minib0_deconv_sm5'] = df['Ifast_minib0_deconv'].rolling(window=3, center=True).mean()


    # plt.legend(loc='upper left')
    # plt.legend(loc='upper left', frameon=True, fontsize='xx-large', markerscale=10, handletextpad=0.1 )
    # plt.legend(loc='upper left', frameon=True, fontsize='medium', markerscale=10, handletextpad=0.1 )
    plt.legend(loc='upper left', frameon=True, fontsize='large', markerscale=10, handletextpad=0.1 )


    # plt.xlim(1700, 8700)

    #
    plt.title(ptitle, fontsize = size_title)

    # plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Check_Plots/beta_checks/{sp}_nolog.png')

    plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v3/Plots_PerSim/png/' + pre + sp + '.png')
    # plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v3/Plots_PerSim/eps/' + pre  + sp + '.eps')
    # plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v3/Plots_PerSim/pdf/' + pre + sp + '.pdf')
    plt.show()

#
#
# fig, ax0 = plt.subplots()
#
# plt.xlabel(r'Time($[sec.]$)')
# df1t = df1t.reset_index(drop=True)
# df1t['i_fast_par_sm10'] = smooth_parabolic(df1t,'Ifast_minib0_deconv',df1t.Tsim, 5 )
# df1t['i_fast_par_sm20'] = smooth_parabolic(df1t,'Ifast_minib0_deconv',df1t.Tsim, 10 )
#
# # plt.plot(df1t.Tsim_original, df1t.Ifast_minib0_deconv_ib1_decay.rolling(10, center= True).mean(), label = 'i fast deconv 10 sec. smoth., ib1 decay')
# # plt.plot(df1t.Tsim_original, df1t.Ifast_minib0_deconv_ib1_decay_presm10, label = 'i fast deconv. 10 sec. pre smoothed')
# # plt.plot(df1t.Tsim_original, df1t.IM - df1t.at[0,'iB0'], label = 'I ECC - iB0')
# plt.plot(df1t.Tsim, df1t.IM - df1t.at[0,'iB0'], label = 'I ECC - iB0 ')
#
# plt.plot(df1t.Tsim, df1t.I_OPM_jma, label = 'I OPM')
# plt.plot(df1t.Tsim, df1t.Ifast_minib0_deconv.rolling(5, center= True).mean(), label = 'i fast deconv rolling 10 sec. smoth.')
# plt.plot(df1t.Tsim, df1t.i_fast_par_sm10, label = 'i fast deconv parabolic 10 sec. smoth.')
# plt.plot(df1t.Tsim, df1t.i_fast_par_sm20, label = 'i fast deconv parabolic 20 sec. smoth.')
#
# plt.xlim(2050, 2450)
# plt.ylim(-0.2, 0.75)
#
# ax0.legend(loc='best', frameon=False, fontsize='small')
# plt.show()
#
#
# fig, ax0 = plt.subplots()
#
# plt.xlabel(r'Time($[sec.]$)')
# df3t = df3t.reset_index(drop=True)
# df3t['Tsim'] = df3t['Tsim_original']
# df3t['i_fast_par_sm10'] = smooth_parabolic(df3t,'Ifast_minib0_deconv',df3t.Tsim, 5 )
# df3t['i_fast_decay_par_sm10'] = smooth_parabolic(df3t,'Ifast_minib0_deconv_ib1_decay_all',df3t.Tsim, 5 )
#
#
# # plt.plot(df3t.Tsim_original, df3t.Ifast_minib0_deconv_ib1_decay.rolling(10, center= True).mean(), label = 'i fast deconv 10 sec. smoth., ib1 decay')
# # plt.plot(df3t.Tsim_original, df3t.Ifast_minib0_deconv_ib1_decay_presm10, label = 'i fast deconv. 10 sec. pre smoothed')
# # plt.plot(df3t.Tsim_original, df3t.IM - df3t.at[0,'iB0'], label = 'I ECC - iB0')
# plt.plot(df3t.Tsim_original, df3t.IM, label = 'I ECC ')
#
# plt.plot(df3t.Tsim_original, df3t.I_OPM_jma, label = 'I OPM')
# plt.plot(df3t.Tsim_original, df3t.Ifast_minib0_deconv.rolling(5, center= True).mean(), label = 'i fast deconv 10 sec. smoth.')
# plt.plot(df3t.Tsim_original, df3t.i_fast_par_sm10, label = 'i fast deconv parabolic 10 sec. smoth.')
# plt.plot(df3t.Tsim_original, df3t.i_fast_decay_par_sm10, label = 'i fast deconv decay parabolic 10 sec. smoth.')
#
#
# ax0.legend(loc='best', frameon=False, fontsize='small')
#
# plt.show()
#
#
