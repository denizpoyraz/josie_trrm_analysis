# Libraries
# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter, MultipleLocator,FormatStrFormatter

from analyse_functions import calc_average_df_pressure, set_columns_nopair_dependence, cal_dif
from constant_variables import *
from data_cuts import cuts0910, cuts2017
from homogenization_functions import return_phipcor
from plotting_functions import filter_rdif_all

bool_sm_vh = True
bool_gs = False
if bool_sm_vh:
    pre = '_sm_hv'
else: pre = ''


df17c = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_deconv_2023paper_HVbeta_noib0.csv", low_memory=False)
df17c = df17c[df17c.iB2 > -9]
df09c = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/"
                    f"Josie0910_deconv_2023_unitedpaper_HVbeta.csv", low_memory=False)


# df09c['Ifast_minib0_deconv_ib1_decay_sm10'] = df09c['Ifast_minib0_deconv_ib1_decay'].rolling(window=5,center=True).mean()
df09c['Ifast_minib0_deconv_sm10'] = df09c['Ifast_deconv_ib1_decay'].rolling(window=5, center=True).mean()
df17c['Ifast_minib0_deconv_sm10'] = df17c['Ifast_deconv'].rolling(window=5, center=True).mean()
if bool_sm_vh:
    df09c['Ifast_minib0_deconv_sm10'] = df09c['Ifast_deconv_ib1_decay']
    df17c['Ifast_minib0_deconv_sm10'] = df17c['Ifast_deconv']
    

df09c['Islow_conv'] = df09c['I_slow_conv_ib1_decay']
df09c['Ifast_minib0_deconv'] = df09c['Ifast_deconv_ib1_decay']
df09c['Ifast_minib0'] = df09c['I_fast_ib1_decay']
df17c['Ifast_minib0'] = df17c['Ifast']
df17c['Ifast_minib0_deconv'] = df17c['Ifast_deconv']

df09 = cuts0910(df09c)
df17 = cuts2017(df17c)

df09['Year'] = '0910'
df17['Year'] = '2017'

dfa = pd.concat([df09, df17], ignore_index=True)
dfa = dfa[dfa.iB0 < 1]

# year = '2009,2010,2017'
year = '0910'
tyear = '2009/2010'
# year = '2017'
# tyear = '2017'

df = dfa[dfa.Year == year]

#
prof = filter_rdif_all(df)
# # prof = [profEN0505, profEN1010, profEN1001, profSP0505, profSP1010, profSP1001]
beta_l = [beta_en0505,beta_en1010,beta_1001, beta_sp0505,beta_sp1010, beta_1001]
# 0910
ilist = [0,1,3,4]
if year == '2017':ilist = [0,2,4,5]

for i in ilist:
    betac = beta_l[i]
    dv = 100
    ac = a[i]/dv
    bc = b[i]/dv
    ac_err = a_err[i]/dv
    bc_err = b_err[i]/dv
    #
    dfi = prof[i]
    dfi['iB0'] = 0
    dfi['iB2'] = 0

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
        slist = ['PO3_dqa', 'PO3_OPM', 'PO3_trrm', 'PO3', 'IM','Ifast_minib0_deconv_sm10','Islow_conv',
                 'Ifast_minib0_deconv','Ifast_minib0']
        clist = ['PO3_dqac', 'PO3_OPMc', 'PO3_trrmc','PO3',  'IMc','Ifast_minib0_deconv_sm10c','Islow_convc',
                 'Ifast_minib0_deconvc', 'Ifast_minib0c']
        vlist = ['PFcor', 'iB0', 'iB1', 'iB2','Pair', 'ENSCI', 'Sol', 'Buf', 'Sim', 'Team',
                 'Tsim', 'TPext', 'Tpump_cor',  'Cpf_kom', 'Cpf_jma', 'unc_Cpf_kom', 'unc_Cpf_jma', 'PFcor_kom', 'PFcor_jma']

        for j in range(len(slist)):
            dfi[clist[j]] = dfi[slist[j]]
            dfi.loc[(dfi.Pair < 500) & (dfi.Pair > 350) & (dfi.ENSCI == 1), clist[j]] = np.nan
            dfi.loc[(dfi.Pair < 600) & (dfi.Pair > 300) & (dfi.ENSCI == 0), clist[j]] = np.nan
            dfi.loc[(dfi.Pair < 120) & (dfi.Pair > 57), clist[j]] = np.nan
            dfi.loc[(dfi.Pair < 30) & (dfi.Pair > 7), clist[j]] = np.nan

        dfs = dfi[clist].interpolate()
        for v in vlist:
            dfs[v] = dfi[v]

        dfi = dfs.copy()
        for j in range(len(slist)):
            dfi[slist[j]] = 0
            dfi[slist[j]] = dfs[clist[j]]

    column_list = ['Pair','Tsim','IM','Ifast_minib0_deconv_sm10','Islow_conv','TPext', 'Tpump_cor','PO3','PO3_dqa',
                       'PO3_trrm','Cpf_kom', 'Cpf_jma','unc_Cpf_kom','unc_Cpf_jma','PO3_OPM', 'Ifast_minib0_deconv','Ifast_minib0']


    df = calc_average_df_pressure(dfi, column_list, yref)
    nop_columns = ['PFcor', 'iB0', 'iB1', 'iB2','PFcor_kom', 'PFcor_jma']
    df = set_columns_nopair_dependence(dfi, df,  nop_columns)


    # df['ratio_2'] = -df['iB0']/df['IM']*100
    df['ratio_3'] = -df['Islow_conv']/df['IM']*100
    df['ratio_6'] = (df['Ifast_minib0_deconv'] - df['Ifast_minib0'])/df['IM']*100
    # df['ratio_7'] = (df['PFcor_kom'] - df['PFcor_jma']) / df['PFcor_jma']
    df['ratio_7'] = (df['Cpf_jma']-1)*100

    df['ratio_8'] =  df['ratio_3'] + df['ratio_6'] + df['ratio_7']

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
    df['d_pfe_hum_trrm'] = ((df['unc_Cpf_jma'] / df['Cpf_jma']) ** 2) + \
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
        np.sqrt(df['d_im_bkg_trm'] + df['d_pfe_hum_trrm'] + df['d_eta_c_trm'] + df['d_eta_a'] + df[
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
    df['d_pfe_hum_trrm'] = np.sqrt(((df['unc_Cpf_jma'] / df['Cpf_jma']) ** 2) + \
                              ((df['dPhim'] * df['dPhim'] + unc_cPL ** 2 + unc_cPH ** 2)))
    # c
    df['d_eta_c'] = (df['deta_c'] / df['eta_c'])
    df['d_eta_c_trm'] = np.sqrt(((ac_err ** 2 + (np.log10(df['Pair']) * bc_err) ** 2) / (1 + ac + bc * np.log10(df['Pair'])) ** 2) + opm_err ** 2)

    # d
    df['d_eta_a'] = (df['deta_a'] / df['eta_a'])
    # e
    df['d_tpump'] = (df['dtpump'] / df['Tpump_cor'])
    ################################################################################################################

   #plotting

    xrtitle = 'Relative uncertainty [%]'
    xptitle = 'Ozone partial pressure [mPa]'
    ytitle = 'Pressure [hPa]'
    xptitle2 = 'O3 [mPa]'

    ###plotting
    trrm_name = f'unc_{pre}_{year}_TRC_{labellist[i]}.png'
    print(trrm_name)
    size_label =22
    size_title =24

    title = f'{tyear} {labellist[i]}'
    titleo = f'{tyear}'
    ###fig 2 trm
    fig, ax = plt.subplots(figsize=(11,20))
    plt.yscale('log')
    ax.set_title(titleo, fontsize=size_title)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(which='major', width=2)
    ax.tick_params(which='minor', width=2)
    plt.gca().tick_params(which='major', width=2)
    plt.gca().tick_params(which='minor', width=2)
    # plt.gca().xaxis.set_minor_locator(MultipleLocator(0.025))
    plt.gca().xaxis.set_tick_params(length=5, which='minor')
    plt.gca().xaxis.set_tick_params(length=10, which='major')
    plt.gca().yaxis.set_tick_params(length=5, which='minor')
    plt.gca().yaxis.set_tick_params(length=10, which='major')
    plt.ylim([1000, 7])

    if year == '0910':
        plt.ylim([1000, 5])

    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))

    # ax.set_yticklabels([])
    # plt.xticks(np.arange(-0.05, 0.19, 0.05))

    # plt.ylabel(ytitle, fontsize=size_label)
    plt.yticks(fontsize=size_label)
    plt.ylabel(ytitle, fontsize=size_label)
    plt.xticks(fontsize=size_label)
    plt.xlabel(xptitle, fontsize=size_label)
    plt.plot(df['PO3_OPM'], df['Pair'], color=cbl[8], label=f'OPM', linewidth=3)
    # plt.plot(df['PO3_trrm'], df['Pair'], color=cbl[i], label=f'TRC', linewidth=3)
    # plt.xlim([0, 22])

    # ax.legend(loc='best', frameon=True, fontsize='xx-large')
    ax.legend(loc='best', frameon=True, fontsize='xx-large', markerscale=1, handletextpad=0.1)
    plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/HVbeta_noib0/Total_Unc/profile_{trrm_name}')

    if not bool_gs:
        fig, ax0 = plt.subplots(figsize=(11,20))
    if bool_gs:
        fig = plt.figure(figsize=(15, 12))
        plt.suptitle(title, fontsize=size_title, y=0.98)
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 2])
        gs.update(wspace=0.05, hspace=0.05)
        ax0 = plt.subplot(gs[0])

    plt.yscale('log')
    ax0.set_title(title, fontsize=size_title)
    if bool_gs:
        ax0.set_title('Contributions', fontsize=size_label)

    ax0.yaxis.set_major_formatter(ScalarFormatter())
    ax0.tick_params(which='major', width=2)
    ax0.tick_params(which='minor', width=2)
    plt.gca().tick_params(which='major', width=2)
    plt.gca().tick_params(which='minor', width=2)
    ax0.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))

    plt.gca().xaxis.set_major_locator(MultipleLocator(5))
    plt.gca().xaxis.set_tick_params(length=5, which='minor')
    plt.gca().xaxis.set_tick_params(length=10, which='major')
    plt.gca().yaxis.set_tick_params(length=5, which='minor')
    plt.gca().yaxis.set_tick_params(length=10, which='major')
    plt.ylim([1000, 5])
    # plt.xlim([-25, 20])

    if year == '0910':
        plt.ylim([1000, 5])
        # plt.xlim([-16, 20])


    plt.yticks(fontsize=size_label)
    plt.ylabel(ytitle, fontsize=size_label)
    plt.xticks(fontsize=size_label)
    # plt.xlabel('Ratio', fontsize=size_label)
    plt.xlabel('Relative corrections [%]', fontsize=size_label)

    # plt.plot(df['ratio_2'], df['Pair'],  marker='o', label=r'I$_{B0}$/I$_M$', markersize=0.5, linewidth=2, color=cbl[1])
    plt.plot(df['ratio_3'], df['Pair'],   marker='o', label=r'I$_S$/I$_M$', markersize=0.5, linewidth=3, color=cbl[2])
    plt.plot(df['ratio_6'], df['Pair'],   marker='o', label=r'(I$_{F,D,S}$ - I$_F$)/I$_M$', markersize=0.5,linewidth=2, color=cbl[3])
    plt.plot(df['ratio_7'], df['Pair'],   marker='o', label=r'(1/JMA - 1)', markersize=0.5,linewidth=2, color=cbl[4])
    # plt.plot(df['ratio_7'], df['Pair'],   marker='o', label=r'($\eta_{p,kom} -\eta_{p,jma})/\eta_{p,kom}$', markersize=0.5,linewidth=2, color=cbl[4])

    plt.plot(df['ratio_8'], df['Pair'],   marker='o', label=r'Total', markersize=0.5,linewidth=2, color=cbl[0])
    # plt.plot(df['ratio_2'], df['Pair'], marker='o', label=r'(I$_M$ - I$_{B0}$)/I$_M$', markersize=0.5, linewidth=2, color=cbl[1])
    # plt.plot(df['ratio_3'], df['Pair'], marker='o', label=r'(I$_M$ - I$_S$)/I$_M$', markersize=0.5, linewidth=3, color=cbl[2])
    # plt.plot(df['ratio_6'], df['Pair'], marker='o', label=r'(I$_M$ - I$_{F,D,S}$ - I$_F$)/I$_M$', markersize=0.5, linewidth=2,
    #          color=cbl[3])
    # plt.plot(df['ratio_7'], df['Pair'], marker='o', label=r'(I$_M$ -1 - 1/JMA)/I$_M$',
    #          markersize=0.5, linewidth=2, color=cbl[4])
    # plt.plot(df['ratio_8'], df['Pair'], marker='o', label=r'Total', markersize=0.5, linewidth=2, color=cbl[0])

    ax0.legend(loc='best', frameon=True, fontsize='xx-large', markerscale=1,  handletextpad=0.1)
    if not bool_gs:
        plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/HVbeta_noib0/Total_Unc/relative_corrections_{trrm_name}')
        # plt.show()
        # plt.close()
    if not bool_gs:
        fig, ax1 = plt.subplots(figsize=(11,20))
    if bool_gs:
        ax1 = plt.subplot(gs[1])

    plt.yscale('log')
    plt.ylim([1000, 5])

    ax1.set_title(title, fontsize=size_label)
    if bool_gs:
        ax1.set_title('Uncertainty Budget', fontsize=size_label)
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.xaxis.set_major_formatter(ScalarFormatter())

    ax1.tick_params(which='major', width=2)
    ax1.tick_params(which='minor', width=2)
    plt.gca().tick_params(which='major', width=2)
    plt.gca().tick_params(which='minor', width=2)
    # plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
    plt.gca().xaxis.set_tick_params(length=5, which='minor')
    plt.gca().xaxis.set_tick_params(length=10, which='major')
    plt.gca().yaxis.set_tick_params(length=5, which='minor')
    plt.gca().yaxis.set_tick_params(length=10, which='major')
    if not bool_gs:
        plt.ylabel(ytitle, fontsize=size_label)
    if bool_gs:
        ax1.set_yticklabels([])
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))


    plt.yticks(fontsize=size_label)
    plt.xticks(fontsize=size_label)
    plt.xlabel(xrtitle, fontsize=size_label)



    plt.plot(df['d_im_bkg_trm'] * 100, df['Pair'], color=cbl[1], label=f'Current, Bkg',
             linewidth=2)
    plt.plot(df['d_pfe_hum_trrm'] * 100, df['Pair'], color=cbl[2], label=f'Pump flow rate eff.',
             linewidth=3)
    plt.plot(df['d_eta_c_trm'] * 100, df['Pair'], color=cbl[3], label=f'Conversion',
             linewidth=2)
    plt.plot(df['d_eta_a'] * 100, df['Pair'], color=cbl[4], label=f'Absorption', linewidth=2)
    plt.plot(df['d_tpump'] * 100, df['Pair'], color=cbl[5], label=f'Pump temp.', linewidth=2)
    plt.plot(df['tota_unc_trm'] * 100, df['Pair'], color=cbl[0], label=f'Total TRC  ', linewidth=2)
    plt.plot(df['tota_unc'] * 100, df['Pair'], color=cbl[0], label=f'Total conventional', linewidth=2, linestyle="-.")

    ax1.legend(loc='best', frameon=True, fontsize='xx-large', markerscale=1,  handletextpad=0.1)
    if not bool_gs:
        plt.savefig(
            f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/HVbeta_noib0/Total_Unc/uncertainity_budget_{trrm_name}')
    if bool_gs:
        plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/HVbeta_noib0/Total_Unc/all_{trrm_name}')

    # plt.show()
