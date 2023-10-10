import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def table_beta(dft):




    return dft



def filter_solsonde(df):

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

    return profEN0505_nodup, profEN1010_nodup, profSP0505_nodup, profSP1010_nodup





def ratiofunction_beta_new(df, sim, team, categorystr, boolibo, slow):
    ##modifications made for timzescan, i slow conv used for beta calculation will be calculated in this function!

    r1 = [0] * len(sim);
    r2 = [0] * len(sim);
    r3 = [0] * len(sim);
    r4 = [0] * len(sim)
    
    ri1 = [0] * len(sim);
    ri2 = [0] * len(sim);
    ri3 = [0] * len(sim);
    ri4 = [0] * len(sim)

    ro1 = [0] * len(sim);
    ro2 = [0] * len(sim);
    ro3 = [0] * len(sim);
    ro4 = [0] * len(sim)

    rs1 = [0] * len(sim);
    rs2 = [0] * len(sim);
    rs3 = [0] * len(sim);
    rs4 = [0] * len(sim)

    r1mean = np.zeros(len(sim));
    r2mean = np.zeros(len(sim));
    r3mean = np.zeros(len(sim));
    r4mean = np.zeros(len(sim))
    r1std = np.zeros(len(sim));
    r2std = np.zeros(len(sim));
    r3std = np.zeros(len(sim));
    r4std = np.zeros(len(sim))
    r1median = np.zeros(len(sim));
    r2median = np.zeros(len(sim));
    r3median = np.zeros(len(sim));
    r4median = np.zeros(len(sim))
    qerr_r1 = np.zeros(len(sim))
    qerr_r2 = np.zeros(len(sim))
    qerr_r3 = np.zeros(len(sim))
    qerr_r4 = np.zeros(len(sim))

    ri1median = np.zeros(len(sim));
    ri2median = np.zeros(len(sim));
    ri3median = np.zeros(len(sim));
    ri4median = np.zeros(len(sim))

    ro1median = np.zeros(len(sim));
    ro2median = np.zeros(len(sim));
    ro3median = np.zeros(len(sim));
    ro4median = np.zeros(len(sim))

    rs1median = np.zeros(len(sim));
    rs2median = np.zeros(len(sim));
    rs3median = np.zeros(len(sim));
    rs4median = np.zeros(len(sim))

    dft = {}
    df1 = {}
    df2 = {}
    df3 = {}
    df4 = {}

    dftab = pd.DataFrame()


    for j in range(len(sim)):
        # print('simarray', sim[j])
        title = str(sim[j]) + '-' + str(team[j])

        r1_down = 2350 -20 ;
        r1_up = 2400 -20;
        r2_down = 4350 -20;
        r2_up = 4400-20;
        r3_down = 6350-20;
        r3_up = 6400-20;
        # r4_down = 8350 -20;
        # r4_up = 8400 -20
        #test1
        r4_down = 8350 -50;
        r4_up = 8400 -50

        if sim[j] == 140:
            r1_down = 2700;
            r1_up = 2740;
            r2_down = 4700;
            r2_up = 4740;
            r3_down = 6700;
            r3_up = 6740;
            r4_down = 8700;
            r4_up = 8740

        if sim[j] == 166:
            r1_down = 2350 + 180;
            r1_up = 2400 + 180;
            r2_down = 4350 + 180;
            r2_up = 4400 + 180;
            r3_down = 6350 + 180;
            r3_up = 6400 + 180;
            r4_down = 8350 + 180;
            r4_up = 8400 + 180

        if sim[j] == 161:
            r1_down = 2350 + 30;
            r1_up = 2400 + 30;
            r2_down = 4350+ 30;
            r2_up = 4400 + 30;
            r3_down = 6350 + 30;
            r3_up = 6400 + 30;
            r4_down = 8350;
            r4_up = 8400

        if sim[j] == 162:
            r1_down = 2350 +30;
            r1_up = 2400 +30;
            r2_down = 4350+30;
            r2_up = 4400 +30;
            r3_down = 6350 +30;
            r3_up = 6400 +30;
            r4_down = 8350 +30;
            r4_up = 8400 +30

        if (sim[j] == 143):
            # | (sim[j] == 161)
            r4_down = 8350 + 60;
            r4_up = 8400 + 60
        # if (sim[j] == 137) | (sim[j] == 141) | (sim[j] == 149):

        # if sim[j] != 166: continue

        dft[j] = df[(df.Sim == sim[j]) & (df.Team == team[j])]
        dft[j].reset_index(inplace=True)

        size = len(dft[j])
        Ums_i = [0] * size
        Ua_i = [0] * size

        Ums_i[0] = dft[j].at[0, 'IM']

        # dft[j]['IMminusiB0'] = dft[j]['IM'] - dft[j]['iB0']

        ## only convolute slow part of the signal, which is needed for beta calculation
        for i in range(size - 1):
            # Ua_i = dft[j].at[i+1, 'IMminusiB0']
            Ua_i[i] = dft[j].at[i, 'I_OPM_jma']
            Ua_i[i+1] = dft[j].at[i + 1, 'I_OPM_jma']
            t1 = dft[j].at[i + 1, 'Tsim']
            t2 = dft[j].at[i, 'Tsim']
            Xs = np.exp(-(t1 - t2) / slow)
            Ums_i[i + 1] = Ua_i[i + 1] - (Ua_i[i + 1] - Ums_i[i]) * Xs
        # fi = dft[j].first_valid_index()
        # li = dft[j].last_valid_index()
        dft[j].loc[:, 'I_conv_slow_jma'] = Ums_i
        # dft[j]['I_conv_slow_jma'] = Ums_i


        df1[j] = dft[j][(dft[j].Tsim >= r1_down) & (dft[j].Tsim < r1_up)]
        df2[j] = dft[j][(dft[j].Tsim >= r2_down) & (dft[j].Tsim < r2_up)]
        df3[j] = dft[j][(dft[j].Tsim >= r3_down) & (dft[j].Tsim < r3_up)]
        df4[j] = dft[j][(dft[j].Tsim >= r4_down) & (dft[j].Tsim < r4_up)]

        if not boolibo:
            r1[j] = np.array((df1[j].IM / (df1[j].I_conv_slow_jma)).tolist())
            r2[j] = np.array((df2[j].IM / (df2[j].I_conv_slow_jma)).tolist())
            r3[j] = np.array((df3[j].IM / (df3[j].I_conv_slow_jma)).tolist())
            r4[j] = np.array((df4[j].IM / (df4[j].I_conv_slow_jma)).tolist())

        if boolibo:
            r1[j] = np.array(((df1[j].IM - df1[j].iB0) / (df1[j].I_conv_slow_jma)).tolist())
            r2[j] = np.array(((df2[j].IM - df2[j].iB0) / (df2[j].I_conv_slow_jma)).tolist())
            r3[j] = np.array(((df3[j].IM - df3[j].iB0) / (df3[j].I_conv_slow_jma)).tolist())
            r4[j] = np.array(((df4[j].IM - df4[j].iB0) / (df4[j].I_conv_slow_jma)).tolist())

            ri1[j] = np.array(df1[j].IM - df1[j].iB0)
            ri2[j] = np.array(df2[j].IM - df2[j].iB0)
            ri3[j] = np.array(df3[j].IM - df3[j].iB0)
            ri4[j] = np.array(df4[j].IM - df4[j].iB0)

            ro1[j] = np.array(df1[j].I_OPM_jma)
            ro2[j] = np.array(df2[j].I_OPM_jma)
            ro3[j] = np.array(df3[j].I_OPM_jma)
            ro4[j] = np.array(df4[j].I_OPM_jma)
            rs1[j] = np.array(df1[j].I_conv_slow_jma)
            rs2[j] = np.array(df2[j].I_conv_slow_jma)
            rs3[j] = np.array(df3[j].I_conv_slow_jma)
            rs4[j] = np.array(df4[j].I_conv_slow_jma)

        
        # print(j, np.nanmean(r1[j]))
        r1median[j] = np.nanmedian(r1[j])
        r2median[j] = np.nanmedian(r2[j])
        r3median[j] = np.nanmedian(r3[j])
        r4median[j] = np.nanmedian(r4[j])

        ri1median[j] = np.nanmedian(ri1[j])
        ri2median[j] = np.nanmedian(ri2[j])
        ri3median[j] = np.nanmedian(ri3[j])
        ri4median[j] = np.nanmedian(ri4[j])
        
        ro1median[j] = np.nanmedian(ro1[j])
        ro2median[j] = np.nanmedian(ro2[j])
        ro3median[j] = np.nanmedian(ro3[j])
        ro4median[j] = np.nanmedian(ro4[j])

        rs1median[j] = np.nanmedian(rs1[j])
        rs2median[j] = np.nanmedian(rs2[j])
        rs3median[j] = np.nanmedian(rs3[j])
        rs4median[j] = np.nanmedian(rs4[j])

        # print('medians in the function', r1median[j], r2median[j], r3median[j], r4median[j])

        dftab.loc[j,'Sim'] = title
        dftab.loc[j,'iB0'] = dft[j].loc[dft[j].first_valid_index(),'iB0']
        dftab.loc[j,'iB1'] = dft[j].loc[dft[j].first_valid_index(),'iB1']
        dftab.loc[j,'iB1-iB0'] = dft[j].loc[dft[j].first_valid_index(),'iB1'] - dft[j].loc[dft[j].first_valid_index(),'iB0']

        # print(j, 'r1median[j]', r1median[j])

        dftab.loc[j,'beta1'] = r1median[j]
        dftab.loc[j,'beta2'] = r2median[j]
        dftab.loc[j,'beta3'] = r3median[j]
        dftab.loc[j,'beta4'] = r4median[j]

        # print(j, title, 'rmedians',r1median[j], r2median[j], r3median[j], r4median[j])

        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'beta'] = np.median([r1median[j], r2median[j], r3median[j], r4median[j]])
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'beta1'] = r1median[j]
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'beta2'] = r2median[j]
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'beta3'] = r3median[j]
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'beta4'] = r4median[j]

        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'IM-iB0_1'] = ri1median[j]
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'IM-iB0_2'] = ri2median[j]
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'IM-iB0_3'] = ri3median[j]
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'IM-iB0_4'] = ri4median[j]

        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'IOPM_1'] = ro1median[j]
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'IOPM_2'] = ro2median[j]
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'IOPM_3'] = ro3median[j]
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'IOPM_4'] = ro4median[j]

        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'IOPMconv_1'] = rs1median[j]
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'IOPMconv_2'] = rs2median[j]
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'IOPMconv_3'] = rs3median[j]
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'IOPMconv_4'] = rs4median[j]

        # print(j,'beta1', r1[j], r1median[j] )
        # print(j,'beta2', r2[j], r2median[j] )
        # print(j,'beta3', r3[j], r3median[j] )
        # print(j,'beta4', r4[j], r4median[j] )
        # print(j, 'final', np.median([r1median[j], r2median[j], r3median[j], r4median[j]]))
        # print(j, 'final two', np.median([r1[j], r2[j], r3[j], r4[j]]))




        err = [0] * len(list(dftab))
        err2 = [0] * len(list(dftab))

        for i, k in zip(list(dftab), range(len(list(dftab)))):
            # err[k] = (np.nanquantile(dftab[i].tolist(), 0.75) - np.nanquantile(dftab[i].tolist(), 0.25)) / (2 * 0.6745)
            # print('tab loop', i, k)
            if k == 0: continue
            err2[k] = stats.median_abs_deviation(dftab[i].tolist())

        # plt.close('all')
        # # if sim[j] == 144:
        # fig, axs = plt.subplots(figsize=(8, 6))
        # plt.plot(np.array(dft[j]['Tsim'] / 60), np.array(dft[j]['I_OPM_jma']), label='I OPM.')
        # plt.plot(np.array(dft[j]['Tsim'] / 60), np.array(dft[j]['I_conv_slow_jma']), label='I slow conv.')
        # plt.plot(np.array(dft[j]['Tsim'] / 60), np.array(dft[j]['IM']), label='I ECC')
        # plt.plot(np.array(dft[j]['Tsim'] / 60), np.array(dft[j]['iB0']), label='iB0')
        # plt.plot(np.array(dft[j]['Tsim'] / 60), np.array(dft[j]['iB1']), label='iB1')
        # plt.plot(np.array(dft[j]['Tsim'] / 60), np.array(dft[j]['iB1'] - dft[j]['iB0']), label='iB1 - iB0')
        # # plt.plot(np.array(dft[j]['time']/60), np.array(dft[j]['I_gen_pr']), label ='I gen. previous')
        # plt.axvline(x=r1_down / 60, color='red', linestyle='--', label='t begin [1]', linewidth=1)
        # plt.axvline(x=r1_up / 60, color='black', linestyle='--', label='t end [1]', linewidth=1)
        # plt.axvline(x=r2_down / 60, color='red', linestyle='--', linewidth=1)
        # plt.axvline(x=r2_up / 60, color='black', linestyle='--', linewidth=1)
        # plt.axvline(x=r3_down / 60, color='red', linestyle='--', linewidth=1)
        # plt.axvline(x=r3_up / 60, color='black', linestyle='--', linewidth=1)
        # plt.axvline(x=r4_down / 60, color='red', linestyle='--', linewidth=1)
        # plt.axvline(x=r4_up / 60, color='black', linestyle='--', linewidth=1)
        # # plt.ylim([-0.5, 10])
        # axs.legend(loc='upper right', frameon=True, fontsize='small')
        # axs.set_yscale('log')
        # # plotname = sondenumber + ' ' + str(exp)
        # plt.title(categorystr + ' ' + title)
        # #
        # path = '/home/poyraden/Analysis/JosieAnalysis/Plots/Beta_0910_Plots/'
        # # plt.savefig(path + categorystr + '_' + title + '_v2.eps')
        # plt.savefig(path + categorystr + '_' + title + '_log.png')
        # # plt.show()
        # #v2 is every gap is 10 seconds earlier
        # plt.close()

        r1mean[j] = np.nanmean(r1[j])
        r1std[j] = np.nanstd(r1[j])
        r2mean[j] = np.nanmean(r2[j])
        r2std[j] = np.nanstd(r2[j])
        r3mean[j] = np.nanmean(r3[j])
        r3std[j] = np.nanstd(r3[j])
        r4mean[j] = np.nanmean(r4[j])
        r4std[j] = np.nanstd(r4[j])

        err2[k] = stats.median_abs_deviation(dftab[i].tolist())

        qerr_r1[j] = stats.median_abs_deviation(r1[j])
        qerr_r2[j] = stats.median_abs_deviation(r2[j])
        qerr_r3[j] = stats.median_abs_deviation(r3[j])
        qerr_r4[j] = stats.median_abs_deviation(r4[j])

    rmean = [r1mean, r2mean, r3mean, r4mean]
    rstd = [r1std, r2std, r3std, r4std]
    # rmedianarray = [r1median, r2median, r3median, r4median]

    rmed1 = np.nanmedian(r1median)
    rmed2 = np.nanmedian(r2median)
    rmed3 = np.nanmedian(r3median)
    rmed4 = np.nanmedian(r4median)

    rmedian = [rmed1, rmed2, rmed3, rmed4]


    # print('rmedian funtion', rmedian)

    qerr_1 = stats.median_abs_deviation(r1median)
    qerr_2 = stats.median_abs_deviation(r2median)
    qerr_3 = stats.median_abs_deviation(r3median)
    qerr_4 = stats.median_abs_deviation(r4median)

    qerr = [qerr_1, qerr_2, qerr_3, qerr_4]


    pathtab = '/home/poyraden/Analysis/JosieAnalysis/csv/0910_'
    dftab.to_csv(pathtab + categorystr + 'beta_cor_upd.csv')

    dftab.loc["Mean"] = dftab.mean()
    dftab.loc["std"] = dftab.std()
    dftab.loc["Median"] = dftab.median()
    dftab.loc['median error'] = err2

    dftab.loc['beta1-4 median'] = np.median(rmedian)
    # print((r1median,rmedian[2],rmedian[3]))
    m14 = np.hstack((r1median, r2median,r3median,r4median)).ravel()
    m24 = np.hstack((r2median,r3median,r4median)).ravel()
    dftab.loc['beta1-4 median error'] = stats.median_abs_deviation(m14)
    dftab.loc['beta2-4 median'] = np.median(rmedian[1:4])
    dftab.loc['beta2-4 median error'] = stats.median_abs_deviation(m24)
    dftab = dftab.round(3)

    # print(dftab)

    pathtab = '/home/poyraden/Analysis/JosieAnalysis/Codes/tables/'
    dftab.to_latex(pathtab + categorystr + 'beta_cor_upd.tex', index = False)
    dftab.to_excel(pathtab + categorystr + 'beta_cor_upd.xlsx')
    # print('end of', categorystr)

    return rmean, rstd, rmedian, qerr, df


def ratiofunction_beta_9602_new(df, sim, team, categorystr, boolib0, slow):

    dftab = pd.DataFrame()

    r1 = [0] * len(sim);
    r2 = [0] * len(sim);

    r1mean = np.zeros(len(sim));
    r2mean = np.zeros(len(sim));
    r1std = np.zeros(len(sim));
    r2std = np.zeros(len(sim));

    r1median = np.zeros(len(sim))
    r2median = np.zeros(len(sim))
    qerr_r1 = np.zeros(len(sim))
    qerr_r2 = np.zeros(len(sim))

    dft = {}
    df1 = {}
    df2 = {}

    # df['iB0'] = 0.014

    Pval_komhyr = np.array([1000, 199, 59, 30, 20, 10, 7, 5])
    komhyr_sp_tmp = np.array([1, 1.007, 1.018, 1.022, 1.032, 1.055, 1.070, 1.092])
    komhyr_en_tmp = np.array([1, 1.007, 1.018, 1.029, 1.041, 1.066, 1.087, 1.124])

    komhyr_sp = [1 / i for i in komhyr_sp_tmp]
    komhyr_en = [1 / i for i in komhyr_en_tmp]


    # print('len sim', len(sim))

    for j in range(len(sim)):
        # print('simarray', sim[j], team[j])
        title = str(sim[j]) + '-' + str(team[j])

        dftj = df[(df.Sim == sim[j]) & (df.Team == team[j])]
        if sim[j] == 92:
            dftj = dftj[dftj.Tsim > 100]
        dftj.reset_index(inplace=True)

        size = len(dftj)
        Ums_i = [0] * size
        Ua_i = [0] * size
        Ums_i[0] = dftj.at[0, 'IM']

        ## only convolute slow part of the signal, which is needed for beta calculation
        for k in range(size - 1):

            ## komhyr corrections
            for p in range(len(komhyr_en) - 1):

                if dftj.at[k, 'ENSCI'] == 1:
                    if (dftj.at[k, 'Pair'] >= Pval_komhyr[p + 1]) & (dftj.at[k, 'Pair'] < Pval_komhyr[p]):
                        # print(p, Pval[p + 1], Pval[p ])
                        dftj.at[k, 'I_OPM_komhyr'] = dftj.at[k, 'PO3_OPM'] * dftj.at[k, 'PFcor'] * komhyr_en[p] / \
                                                     (dftj.at[k, 'TPint'] * 0.043085)
                if dftj.at[k, 'ENSCI'] == 0:
                    if (dftj.at[k, 'Pair'] >= Pval_komhyr[p + 1]) & (dftj.at[k, 'Pair'] < Pval_komhyr[p]):
                        # print(p, Pval[p + 1], Pval[p ])
                        dftj.at[k, 'I_OPM_komhyr'] = dftj.at[k, 'PO3_OPM'] * dftj.at[k, 'PFcor'] * komhyr_sp[p] / \
                                                     (dftj.at[k, 'TPint'] * 0.043085)

            if (dftj.at[k, 'Pair'] <= Pval_komhyr[7]):

                if dftj.at[k, 'ENSCI'] == 1:
                    dftj.at[k, 'I_OPM_komhyr'] = dftj.at[k, 'PO3_OPM'] * dftj.at[k, 'PFcor'] * komhyr_en[7] / \
                                                 (dftj.at[k, 'TPint'] * 0.043085)
                if dftj.at[k, 'ENSCI'] == 0:
                    dftj.at[k, 'I_OPM_komhyr'] = dftj.at[k, 'PO3_OPM'] * dftj.at[k, 'PFcor'] * komhyr_sp[7] / \
                                                 (dftj.at[k, 'TPint'] * 0.043085)

            size = len(dftj)
            Ums_i = [0] * size
            Ua_i = [0] * size
            Ums_i[0] = dftj.at[0, 'IM']

            ## only convolute slow part of the signal, which is needed for beta calculation
        for ik in range(size - 1):
            Ua_i = dftj.at[ik + 1, 'I_OPM_jma']
            t1 = dftj.at[ik + 1, 'Tsim']
            t2 = dftj.at[ik, 'Tsim']
            Xs = np.exp(-(t1 - t2) / slow)
            Ums_i[ik + 1] = Ua_i - (Ua_i - Ums_i[ik]) * Xs

        # print('ISLOW COnv',Ums_i )
        # print('I OPM JMA',dftj[['Tsim','I_OPM_jma']] )
        # print('Ums_i[0]', Ums_i[0])
        dftj.loc[:,'I_conv_slow_jma'] = Ums_i

        year = (dftj.iloc[0]['JOSIE_Nr'])

        rt1 = (dftj.iloc[0]['R1_Tstop']) - 5
        rt2 = (dftj.iloc[0]['R2_Tstop']) - 5
        # print(rt1/1, rt2/1)
        if sim[j] == 90:
            rt1 = rt1 - 100

            # rt2 = rt2 - 100
        t1 = (dftj.Tsim <= rt1) & (dftj.Tsim >= rt1-25)
        t2 = (dftj.Tsim <= rt2) & (dftj.Tsim >= rt2-25)
        #
        df1[j] = dftj[t1]
        df2[j] = dftj[t2]
        # c = df1[j].IM .tolist()
        # sc = ( df1[j].I_conv_slow.tolist())

        if not boolib0:
            r1[j] = np.array((df1[j].IM / (df1[j].I_conv_slow_jma)).tolist())
            r2[j] = np.array((df2[j].IM / (df2[j].I_conv_slow_jma)).tolist())

        if boolib0:
            r1[j] = np.array(((df1[j].IM - df1[j].iB0) / (df1[j].I_conv_slow_jma)).tolist())
            r2[j] = np.array(((df2[j].IM - df2[j].iB0) / (df2[j].I_conv_slow_jma)).tolist())

        # print(sim[j],team[j], 'Ratio', r1[j], r2[j])
        # print(sim[j], team[j], 'Curent', df1[j].IM.tolist(), df1[j].I_conv_slow.tolist() )
        #
        r1mean[j] = np.nanmean(r1[j])
        r2mean[j] = np.nanmean(r2[j])
        r1std[j] = np.nanstd(r1[j])
        r2std[j] = np.nanstd(r2[j])

        r1median[j] = np.nanmedian(r1[j])
        r2median[j] = np.nanmedian(r2[j])
        qerr_r1[j] = stats.median_abs_deviation(r1[j])
        qerr_r2[j] = stats.median_abs_deviation(r2[j])

        dftab.loc[j, 'Sim'] = title
        dftab.loc[j, 'iB0'] = dftj.loc[dftj.first_valid_index(), 'iB0']
        try:
            dftab.loc[j, 'iB1'] = dftj.loc[dftj.first_valid_index(), 'iB1']
        except KeyError:
            dftab.loc[j, 'iB1'] = 0

        dftab.loc[j, 'iB1-iB0'] = dftj.loc[dftj.first_valid_index(), 'iB1'] - dftj.loc[dftj.first_valid_index(), 'iB0']


        if dftab.loc[j, 'iB0'] == -99:
            dftab.loc[j, 'iB0'] = np.nan
            dftab.loc[j, 'iB1-iB0'] = np.nan
        if dftab.loc[j, 'iB1'] == -99:
            dftab.loc[j, 'iB1'] = np.nan
            dftab.loc[j, 'iB1-iB0'] = np.nan


        # print('iB1', dftab.loc[j, 'iB1'], type(dftab.loc[j, 'iB1']))

        dftab.loc[j, 'beta1'] = r1median[j]
        dftab.loc[j, 'beta2'] = r2median[j]

        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'beta'] = np.median([r1median[j], r2median[j]])
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'beta1'] = r1median[j]
        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'beta2'] = r2median[j]


        # print('list dftab', list(dftab))
        # print(dftab)

        # plt.close('all')
        #
        # fig, axs = plt.subplots(figsize=(8, 6))
        # plt.plot(np.array(dftj['Tsim']/1), np.array(dftj['I_OPM_jma']), label='I OPM.')
        # plt.plot(np.array(dftj['Tsim']/1), np.array(dftj['I_conv_slow_jma']), label='I slow conv.')
        # plt.plot(np.array(dftj['Tsim']/1), np.array(dftj['IM']), label='I ECC')
        # plt.plot(np.array(dftj['Tsim']/1), np.array(dftj['iB0']), label='iB0')
        # plt.plot(np.array(dftj['Tsim']/1), np.array(dftj['iB1']), label='iB1')
        # plt.plot(np.array(dftj['Tsim']/1), np.array(dftj['iB1'] - dftj['iB0']), label='iB1 - iB0')
        #
        # # plt.plot(np.array(dft[j]['time']/1), np.array(dft[j]['I_gen_pr']), label ='I gen. previous')
        # plt.axvline(x= (rt1-25)/1, color='red', linestyle='--', label='t begin [1]', linewidth=1)
        # plt.axvline(x=(rt1)/1, color='black', linestyle='--', label='t end [1]', linewidth=1)
        # plt.axvline(x=(rt2-25)/1, color='red', linestyle='--', linewidth=1)
        # plt.axvline(x=rt2/1, color='black', linestyle='--', linewidth=1)
        #
        # # plt.ylim([0.0005, 2])
        # plt.xlim([550, 5300])
        #
        # axs.set_yscale('log')
        # axs.legend(loc='upper right', frameon=True, fontsize='small')
        #
        # # plotname = sondenumber + ' ' + str(exp)
        # plt.title('9602 ' + categorystr + ' ' + title)
        # #
        # path = '/home/poyraden/Analysis/JosieAnalysis/Plots/Beta_9602_Plots/'
        # # # plt.savefig(path  + categorystr + '_' + title + '_v2.eps')
        # plt.savefig(path  + categorystr + '_' + title + '_zoom_log.png')
        # # plt.show()
        # # v2 is every gap is 10 seconds earlier
        # plt.close()

        err2 = [0] * len(list(dftab))

    for i, k in zip(list(dftab), range(len(list(dftab)))):
        # err[k] = (np.nanquantile(dftab[i].tolist(), 0.75) - np.nanquantile(dftab[i].tolist(), 0.25)) / (2 * 0.6745)
        # print('tab loop', i, k)
        dfnan = dftab.copy()
        # print('list dftabj', list(dftab[i]))
        dfnan['ib1_nan'] = 0
        dfnan.loc[dfnan.iB1.isnull(), 'ib1_nan'] = 1
        if k == 0: continue
        err2[k] = stats.median_abs_deviation(dftab[i].tolist())
        # err2[k] = stats.median_abs_deviation(dfnan[dfnan.ib1_nan == 0].tolist())

    pathtab = '/home/poyraden/Analysis/JosieAnalysis/csv/9602_'
    dftab.to_csv(pathtab + categorystr + 'beta_cor_upd.csv')

    # print('err2', err2)
    dftab.loc["Mean"] = dftab.mean()
    dftab.loc["std"] = dftab.std()
    dftab.loc["Median"] = dftab.median()
    dftab.loc['median error'] = err2

    dftab = dftab.round(3)


    pathtab = '/home/poyraden/Analysis/JosieAnalysis/Codes/tables/9602_'
    # dftab = dftab.set_index('Sim')
    dftab.to_latex(pathtab + categorystr + 'beta_cor_upd.tex', index=False)
    # df.style.hide(axis="index").to_latex(hrules=True)
    dftab.to_excel(pathtab + categorystr + 'beta_cor_upd.xlsx')

    rmean = [r1mean, r2mean]
    rstd = [r1std, r2std]

    rmed1 = np.nanmedian(r1median)
    rmed2 = np.nanmedian(r2median)
    rmedian = [rmed1, rmed2]

    # print('rmedian funtion', rmedian)

    qerr_1 = stats.median_abs_deviation(r1median)
    qerr_2 = stats.median_abs_deviation(r2median)

    qerr = [qerr_1, qerr_2]

    return rmean, rstd, rmedian, qerr, df

######
def ratiofunction_beta_9602(df, sim, team, categorystr, boolib0, slow, fast):
    r1 = [0] * len(sim);
    r2 = [0] * len(sim);

    r1mean = np.zeros(len(sim));
    r2mean = np.zeros(len(sim));
    r1std = np.zeros(len(sim));
    r2std = np.zeros(len(sim));

    r1median = np.zeros(len(sim))
    r2median = np.zeros(len(sim))
    qerr_r1 = np.zeros(len(sim))
    qerr_r2 = np.zeros(len(sim))

    dft = {}
    df1 = {}
    df2 = {}

    df['iB0'] = 0.014

    Pval_komhyr = np.array([1000, 199, 59, 30, 20, 10, 7, 5])
    komhyr_sp_tmp = np.array([1, 1.007, 1.018, 1.022, 1.032, 1.055, 1.070, 1.092])
    komhyr_en_tmp = np.array([1, 1.007, 1.018, 1.029, 1.041, 1.066, 1.087, 1.124])

    komhyr_sp = [1 / i for i in komhyr_sp_tmp]
    komhyr_en = [1 / i for i in komhyr_en_tmp]

    file = open('../Latex/9602_TimeConstant_' + categorystr + "5secslatex_table.txt", "w")
    file.write(categorystr + '\n')

    # print('len sim', len(sim))

    for j in range(len(sim)):
        # print('simarray', sim[j])
        title = str(sim[j]) + '-' + str(team[j])

        dftj = df[(df.Sim == sim[j]) & (df.Team == team[j])]
        dftj.reset_index(inplace=True)

        size = len(dftj)
        Ums_i = [0] * size
        Ua_i = [0] * size
        Ums_i[0] = dftj.at[0, 'IM']

        ## only convolute slow part of the signal, which is needed for beta calculation
        for k in range(size - 1):

            ## komhyr corrections
            for p in range(len(komhyr_en) - 1):

                if dftj.at[k, 'ENSCI'] == 1:
                    if (dftj.at[k, 'Pair'] >= Pval_komhyr[p + 1]) & (dftj.at[k, 'Pair'] < Pval_komhyr[p]):
                        # print(p, Pval[p + 1], Pval[p ])
                        dftj.at[k, 'I_OPM_komhyr'] = dftj.at[k, 'PO3_OPM'] * dftj.at[k, 'PFcor'] * komhyr_en[p] / \
                                                     (dftj.at[k, 'TPint'] * 0.043085)
                if dftj.at[k, 'ENSCI'] == 0:
                    if (dftj.at[k, 'Pair'] >= Pval_komhyr[p + 1]) & (dftj.at[k, 'Pair'] < Pval_komhyr[p]):
                        # print(p, Pval[p + 1], Pval[p ])
                        dftj.at[k, 'I_OPM_komhyr'] = dftj.at[k, 'PO3_OPM'] * dftj.at[k, 'PFcor'] * komhyr_sp[p] / \
                                                     (dftj.at[k, 'TPint'] * 0.043085)

            if (dftj.at[k, 'Pair'] <= Pval_komhyr[7]):

                if dftj.at[k, 'ENSCI'] == 1:
                    dftj.at[k, 'I_OPM_komhyr'] = dftj.at[k, 'PO3_OPM'] * dftj.at[k, 'PFcor'] * komhyr_en[7] / \
                                                 (dftj.at[k, 'TPint'] * 0.043085)
                if dftj.at[k, 'ENSCI'] == 0:
                    dftj.at[k, 'I_OPM_komhyr'] = dftj.at[k, 'PO3_OPM'] * dftj.at[k, 'PFcor'] * komhyr_sp[7] / \
                                                 (dftj.at[k, 'TPint'] * 0.043085)

            size = len(dftj)
            Ums_i = [0] * size
            Ua_i = [0] * size
            Ums_i[0] = dftj.at[0, 'IM']

            ## only convolute slow part of the signal, which is needed for beta calculation
        for ik in range(size - 1):
            Ua_i = dftj.at[ik + 1, 'I_OPM_jma']
            t1 = dftj.at[ik + 1, 'Tsim']
            t2 = dftj.at[ik, 'Tsim']
            Xs = np.exp(-(t1 - t2) / slow)
            Ums_i[ik + 1] = Ua_i - (Ua_i - Ums_i[ik]) * Xs


        dftj.loc[:,'I_conv_slow_jma'] = Ums_i

        year = (dftj.iloc[0]['JOSIE_Nr'])

        rt1 = (dftj.iloc[0]['R1_Tstop'])
        rt2 = (dftj.iloc[0]['R2_Tstop'])
        if sim[j] == 90: rt1 = 1050
        t1 = (dftj.Tsim <= rt1) & (dftj.Tsim >= rt1 - 5)
        t2 = (dftj.Tsim <= rt2) & (dftj.Tsim >= rt2 - 5)
        #
        df1[j] = dftj[t1]
        df2[j] = dftj[t2]
        # c = df1[j].IM .tolist()
        # sc = ( df1[j].I_conv_slow.tolist())

        if not boolib0:
            r1[j] = np.array((df1[j].IM / (0.10 * df1[j].I_conv_slow_jma)).tolist())
            r2[j] = np.array((df2[j].IM / (0.10 * df2[j].I_conv_slow_jma)).tolist())

        if boolib0:
            r1[j] = np.array(((df1[j].IM - df1[j].iB0) / (0.10 * df1[j].I_conv_slow_jma)).tolist())
            r2[j] = np.array(((df2[j].IM - df2[j].iB0) / (0.10 * df2[j].I_conv_slow_jma)).tolist())

        # print(sim[j],team[j], 'Ratio', r1[j], r2[j])
        # print(sim[j], team[j], 'Curent', df1[j].IM.tolist(), df1[j].I_conv_slow.tolist() )
        #
        r1mean[j] = np.nanmean(r1[j])
        r2mean[j] = np.nanmean(r2[j])
        r1std[j] = np.nanstd(r1[j])
        r2std[j] = np.nanstd(r2[j])
        r1median[j] = np.nanmedian(r1[j])
        r2median[j] = np.nanmedian(r2[j])
        qerr_r1[j] = (np.nanquantile(r1[j], 0.75) - np.nanquantile(r1[j], 0.25)) / (2 * 0.6745)
        qerr_r2[j] = (np.nanquantile(r2[j], 0.75) - np.nanquantile(r2[j], 0.25)) / (2 * 0.6745)

        lr1 = str(round(r1mean[j], 2)) + '\pm ' + str(round(r1std[j], 2))
        lr2 = str(round(r2mean[j], 2)) + '\pm ' + str(round(r2std[j], 2))
        lr3 = str(round(r1median[j], 2)) + '\pm ' + str(round(qerr_r1[j], 2))
        lr4 = str(round(r2median[j], 2)) + '\pm ' + str(round(qerr_r2[j], 2))

        mat = '$'

        file.write(mat + str(int(
            year)) + '-' + title + mat + ' & ' + mat + lr1 + mat + ' & ' + mat + lr2 + mat + ' & ' + mat + lr3 + mat +
                   ' & ' + mat + lr4 + mat + r'\\' + '\n')

    rmean = [r1mean, r2mean]
    rstd = [r1std, r2std]
    rmedian = [r1median, r2median]
    rqerr = [qerr_r1, qerr_r2]

    qerr_1 = (np.nanquantile(r1median, 0.75) - np.nanquantile(r1median, 0.25)) / (2 * 0.6745)
    qerr_2 = (np.nanquantile(r2median, 0.75) - np.nanquantile(r2median, 0.25)) / (2 * 0.6745)

    mederr = (np.nanquantile(rmedian, 0.75) - np.nanquantile(rmedian, 0.25)) / (2 * 0.6745)

    file.write('\hline' + '\n')
    file.write('\hline' + '\n')
    file.write(
        'Mean & ' + mat + str(round(np.nanmean(r1mean), 2)) + '\pm ' + str(round(np.nanstd(r1mean), 2)) + mat + ' & ' +
        mat + str(round(np.nanmean(r2mean), 2)) + '\pm ' + str(round(np.nanstd(r2mean), 2)) + mat + r'\\' + '\n')
    file.write(
        'Median  & ' + mat + str(round(np.nanmedian(r1median), 2)) + '\pm ' + str(round(qerr_1, 2)) + mat + ' & ' +
        mat + str(round(np.nanmedian(r2median), 2)) + '\pm ' + str(round(qerr_2, 2)) + mat + r'\\' + '\n')
    file.write('\hline' + '\n')
    file.write('\hline' + '\n')
    file.write('Mean R1-R2 &' + mat + str(round(np.nanmean(rmean), 2)) + '\pm ' + str(
        round(np.nanstd(rmean), 2)) + mat + ' & ' + r'\\' + '\n')
    file.write('Median R1-R2 &' + mat + str(round(np.nanmedian(rmedian), 2)) + '\pm ' + str(
        round(mederr, 2)) + mat + ' & ' + r'\\' + '\n')

    file.close()

    return rmean, rstd, rmedian, rqerr


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


def ratiofunction_beta_pre(df, sim, team, categorystr, boolibo, slow, fast):
    ##modifications made for timzescan, i slow conv used for beta calculation will be calculated in this function!

    r1 = [0] * len(sim);
    r2 = [0] * len(sim);
    r3 = [0] * len(sim);
    r4 = [0] * len(sim)

    r1mean = np.zeros(len(sim));
    r2mean = np.zeros(len(sim));
    r3mean = np.zeros(len(sim));
    r4mean = np.zeros(len(sim))
    r1std = np.zeros(len(sim));
    r2std = np.zeros(len(sim));
    r3std = np.zeros(len(sim));
    r4std = np.zeros(len(sim))
    r1median = np.zeros(len(sim));
    r2median = np.zeros(len(sim));
    r3median = np.zeros(len(sim));
    r4median = np.zeros(len(sim))
    qerr_r1 = np.zeros(len(sim))
    qerr_r2 = np.zeros(len(sim))
    qerr_r3 = np.zeros(len(sim))
    qerr_r4 = np.zeros(len(sim))

    dft = {}
    df1 = {}
    df2 = {}
    df3 = {}
    df4 = {}

    # df['iB0'] = 0.014

    file = open('../Latex/0910_TimeConstant_beta0' + categorystr + "_preparationdata_table_.txt", "w")
    file.write(categorystr + '\n')
    file.write('\hline' + '\n')

    for j in range(len(sim)):
        # print('simarray', sim[j])
        title = str(sim[j]) + '-' + str(team[j])

        r1_down = 2350;
        r1_up = 2400;
        r2_down = 4350;
        r2_up = 4400;
        r3_down = 6350;
        r3_up = 6400;
        r4_down = 8350;
        r4_up = 8400

        if sim[j] == 140:
            r1_down = 2700;
            r1_up = 2740;
            r2_down = 4700;
            r2_up = 4740;
            r3_down = 6700;
            r3_up = 6740;
            r4_down = 8700;
            r4_up = 8740

        dft[j] = df[(df.Sim == sim[j]) & (df.Team == team[j])]
        dft[j].reset_index(inplace=True)

        # t1 = (dft[j].Tsim >= r1_down) & (dft[j].Tsim < r1_up)
        # t2 = (dft[j].Tsim >= r2_down) & (dft[j].Tsim < r2_up)
        # t3 = (dft[j].Tsim >= r3_down) & (dft[j].Tsim < r3_up)
        # t4 = (dft[j].Tsim >= r4_down) & (dft[j].Tsim < r4_up)

        # zeroindex = int(np.mean(dft[j] .index))

        size = len(dft[j])
        Ums_i = [0] * size
        # Ua_i = [0] * size

        dft[j]['IMminusiB0'] = dft[j]['IM'] - dft[j]['iB0']

        Ums_i[0] = dft[j].at[0, 'IMminusiB0']

        ## only convolute slow part of the signal, which is needed for beta calculation
        for i in range(size - 1):
            # Ua_i = dft[j]['I_OPM_jma'].iloc[i + 1, ]
            Ua_i = dft[j].at[i + 1, 'IMminusiB0']
            t1 = dft[j].at[i + 1, 'Tsim']
            t2 = dft[j].at[i, 'Tsim']
            Xs = np.exp(-(t1 - t2) / slow)
            Xf = np.exp(-(t1 - t2) / fast)
            Ums_i[i + 1] = Ua_i - (Ua_i - Ums_i[i]) * Xs

            # Islow_conv[i + 1] = Islow[i + 1] - (Islow[i + 1] - Islow_conv[i]) * Xs

        dft[j]['I_conv_slow_jma'] = Ums_i

        df1[j] = dft[j][(dft[j].Tsim_original >= r1_down) & (dft[j].Tsim_original < r1_up)]
        df2[j] = dft[j][(dft[j].Tsim_original >= r2_down) & (dft[j].Tsim_original < r2_up)]
        df3[j] = dft[j][(dft[j].Tsim_original >= r3_down) & (dft[j].Tsim_original < r3_up)]
        df4[j] = dft[j][(dft[j].Tsim_original >= r4_down) & (dft[j].Tsim_original < r4_up)]

        if not boolibo:
            r1[j] = np.array((df1[j].IM / (0.10 * df1[j].I_conv_slow_jma)).tolist())
            r2[j] = np.array((df2[j].IM / (0.10 * df2[j].I_conv_slow_jma)).tolist())
            r3[j] = np.array((df3[j].IM / (0.10 * df3[j].I_conv_slow_jma)).tolist())
            r4[j] = np.array((df4[j].IM / (0.10 * df4[j].I_conv_slow_jma)).tolist())

        if boolibo:
            r1[j] = np.array(((df1[j].IM - df1[j].iB0) / (0.10 * df1[j].I_conv_slow_jma)).tolist())
            r2[j] = np.array(((df2[j].IM - df2[j].iB0) / (0.10 * df2[j].I_conv_slow_jma)).tolist())
            r3[j] = np.array(((df3[j].IM - df3[j].iB0) / (0.10 * df3[j].I_conv_slow_jma)).tolist())
            r4[j] = np.array(((df4[j].IM - df4[j].iB0) / (0.10 * df4[j].I_conv_slow_jma)).tolist())
        # print(j, np.nanmean(r1[j]))

        r1mean[j] =1* np.nanmean(r1[j])
        r1std[j] =1* np.nanstd(r1[j])
        r2mean[j] =1* np.nanmean(r2[j])
        r2std[j] =1* np.nanstd(r2[j])
        r3mean[j] =1* np.nanmean(r3[j])
        r3std[j] =1* np.nanstd(r3[j])
        r4mean[j] =1* np.nanmean(r4[j])
        r4std[j] =1* np.nanstd(r4[j])

        r1median[j] = np.nanmedian(r1[j])
        r2median[j] = np.nanmedian(r2[j])
        r3median[j] = np.nanmedian(r3[j])
        r4median[j] = np.nanmedian(r4[j])

        qerr_r1[j] = (np.nanquantile(r1[j], 0.75) - np.nanquantile(r1[j], 0.25)) / (2 * 0.6745)
        qerr_r2[j] = (np.nanquantile(r2[j], 0.75) - np.nanquantile(r2[j], 0.25)) / (2 * 0.6745)
        qerr_r3[j] = (np.nanquantile(r3[j], 0.75) - np.nanquantile(r3[j], 0.25)) / (2 * 0.6745)
        qerr_r4[j] = (np.nanquantile(r4[j], 0.75) - np.nanquantile(r4[j], 0.25)) / (2 * 0.6745)

        lr1 = str(round(r1mean[j], 2)) + '\pm ' + str(round(r1std[j], 2))
        lr2 = str(round(r2mean[j], 2)) + '\pm ' + str(round(r2std[j], 2))
        lr3 = str(round(r3mean[j], 2)) + '\pm ' + str(round(r3std[j], 2))
        lr4 = str(round(r4mean[j], 2)) + '\pm ' + str(round(r4std[j], 2))
        lr5 = str(round(r1median[j], 2)) + '\pm ' + str(round(qerr_r1[j], 2))
        lr6 = str(round(r2median[j], 2)) + '\pm ' + str(round(qerr_r2[j], 2))
        lr7 = str(round(r3median[j], 2)) + '\pm ' + str(round(qerr_r3[j], 2))
        lr8 = str(round(r4median[j], 2)) + '\pm ' + str(round(qerr_r4[j], 2))

        mat = '$'
        end = '&'

        # file.write( mat + title + mat + end +  mat + lr1 + mat + ' & ' + mat + lr2 + mat + ' & ' + mat + lr3 + mat +' & ' + mat + lr4 + mat + r'\\' + '\n')
        # # file.write( mat + title + mat + end + 'Median' + end + mat + lr5 + mat + ' & ' + mat + lr6 + mat + ' & ' + mat + lr7 + mat +' & ' + mat + lr8 + mat + r'\\' + '\n')

    # fig, ax = plt.subplots()
    # fig, axs = plt.subplots(2, 2)
    # fig.suptitle(categorystr)
    # fig.supxlabel('beta0 values')

    # plt.set_title('beta0')

    fig, axs = plt.subplots(2, 2, sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    (ax1, ax2), (ax3, ax4) = axs
    fig.suptitle(categorystr)

    ax1.hist(r1mean, bins=len(r1mean), ls='dashed', color='blue', label='R1')
    # axs[0, 0].set_title('R1')
    ax1.legend(loc='upper right', frameon=False, fontsize='small')
    ax2.hist(r2mean, bins=len(r2mean), ls='dashed', color='blue', label='R2')
    ax2.legend(loc='upper right', frameon=False, fontsize='small')
    ax3.hist(r3mean, bins=len(r3mean), ls='dashed', color='blue', label='R3')
    ax3.legend(loc='upper right', frameon=False, fontsize='small')
    ax4.hist(r4mean, bins=len(r4mean), ls='dashed', color='blue', label='R4')
    ax4.legend(loc='upper right', frameon=False, fontsize='small')

    # ax.legend(loc='upper right', frameon=False, fontsize='small')

    # # plt.hist([r1mean,r2mean,r3mean,r4mean] ,bins=len(r1mean)*2,  label=['R1','R2','R3','R4'],alpha = 0.6, edgeColor = 'black',  histtype='bar', stacked=True, density = 'True')
    #
    # plt.hist([r1median,r2median,r3median,r4median] ,bins=len(r1mean)*2,  label=['R1','R2','R3','R4'],alpha = 0.6, edgeColor = 'black',  histtype='bar', stacked=True, density = 'True')
    #
    # # ax.hist(r2mean, bins=20, ls='dashed', color='blue', label='R2', alpha = 0.6, lw=3, edgeColor = 'black',  histtype='bar', stacked=True)
    # # plt.hist(r3mean, bins=20, color='green', label='R3', alpha = 0.6, ls='dotted', edgeColor = 'black',  histtype='bar', stacked=True)
    # # plt.hist(r4mean, bins=20, color='red', label='R4', alpha=0.6, ls='dashdot', edgeColor = 'black',  histtype='bar', stacked=True)
    #
    # # plt.xlim([-0.1, 1])
    #
    # plt.title(categorystr)
    # plt.xlabel('beta0 values')
    # ax.legend(loc='upper right', frameon=False, fontsize='small')
    plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Beta0/Prev2_'+ categorystr + '.png')
    plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Beta0/Prev2_'+ categorystr + '.eps')
    #
    # plt.show()
    # plt.close()

    rmean = [r1mean, r2mean, r3mean, r4mean]
    rstd = [r1std, r2std, r3std, r4std]
    rmedian = [r1median, r2median, r3median, r4median]

    qerr_1 = (np.nanquantile(r1median, 0.75) - np.nanquantile(r1median, 0.25)) / (2 * 0.6745)
    qerr_2 = (np.nanquantile(r2median, 0.75) - np.nanquantile(r2median, 0.25)) / (2 * 0.6745)
    qerr_3 = (np.nanquantile(r3median, 0.75) - np.nanquantile(r3median, 0.25)) / (2 * 0.6745)
    qerr_4 = (np.nanquantile(r4median, 0.75) - np.nanquantile(r4median, 0.25)) / (2 * 0.6745)

    qerr = [qerr_1, qerr_2, qerr_3, qerr_4]

    mederr = (np.nanquantile(rmedian[1:4], 0.75) - np.nanquantile(rmedian[1:4], 0.25)) / (2 * 0.6745)

    print('median error', mederr)

    print('qerr', qerr)

    # print('Test mean', len(rmean[0]))
    # print('Test median', len(rmedian[0]))

    #     file.write('\hline' + '\n')
    #     file.write('\hline' + '\n')
    #     file.write(
    #         'Mean & ' + mat + str(round(np.nanmean(r1mean), 2)) + '\pm ' + str(round(np.nanstd(r1mean), 2)) + mat + ' & ' +
    #         mat + str(round(np.nanmean(r2mean), 2)) + '\pm ' + str(round(np.nanstd(r2mean), 2)) + mat + ' & ' +
    #         mat + str(round(np.nanmean(r3mean), 2)) + '\pm ' + str(round(np.nanstd(r3mean), 2)) + mat + ' & ' +
    #         mat + str(round(np.nanmean(r4mean), 2)) + '\pm ' + str(round(np.nanstd(r4mean), 2)) + mat +  r'\\' + '\n')
    #     # file.write('Median  & ' + mat + str(np.round(np.nanmedian(r1median), 2)) + '\pm ' + str(np.round(qerr_r1, 2)) + mat + ' & ')
    #     # # +
    #     # #            mat + str(round(np.nanmedian(r2median), 2)) + '\pm ' + str(round(qerr_r2, 2)) + mat +
    #     # #            str(round(np.nanmedian(r3median), 2)) + '\pm ' + str(round(qerr_r3, 2)) + mat + ' & ' + mat +
    #     # #            str(round(np.nanmedian(r4median), 2)) + '\pm ' + str(round(qerr_r4, 2)) + mat + r'\\' + '\n')
    #     # # file.write('\hline' + '\n')
    #     file.write('Median & ' + mat + str(round(np.nanmedian(r1median), 2)) + '\pm ' + str(round(qerr_1, 2)) + mat + ' & ' +
    # mat + str(round(np.nanmedian(r2median), 2)) + '\pm ' + str(round(qerr_2, 2)) + mat + ' & ' +
    #         mat + str(round(np.nanmedian(r3median), 2)) + '\pm ' + str(round(qerr_3, 2)) + mat + ' & ' +
    #         mat + str(round(np.nanmedian(r4median), 2)) + '\pm ' + str(round(qerr_4, 2)) + mat + r'\\' + '\n')
    #
    #     file.write('\hline' + '\n')
    #     file.write('Mean R2-R4 &' + mat + str(round(np.nanmean(rmean[1:4]), 2)) + '\pm ' + str(
    #         round(np.nanstd(rmean[1:4]), 2)) + mat + ' & ' + 'Mean R1-R4 &' + mat + str(round(np.nanmean(rmean), 2)) + '\pm ' + str(
    #         round(np.nanstd(rmean), 2)) + mat + ' & ' +  r'\\' + '\n')
    #     file.write('Median R2-R4 &' + mat + str(round(np.nanmedian(rmedian[1:4]), 2)) + '\pm ' + str(
    #         round(mederr, 2)) + mat + ' & ' + 'Median R1-R4 &' + mat + str(round(np.nanmedian(rmedian), 2)) + '\pm ' + str(
    #         round(np.nanstd(mederr), 2)) + mat + ' & ' +  r'\\' + '\n')

    return rmean, rstd, rmedian, qerr



def ratiofunction_beta_xls(df, sim, team, categorystr, boolibo, slow, fast):
    ##modifications made for timzescan, i slow conv used for beta calculation will be calculated in this function!

    r1 = [0] * len(sim);
    r2 = [0] * len(sim);
    r3 = [0] * len(sim);
    r4 = [0] * len(sim)

    r1mean = np.zeros(len(sim));
    r2mean = np.zeros(len(sim));
    r3mean = np.zeros(len(sim));
    r4mean = np.zeros(len(sim))
    r1std = np.zeros(len(sim));
    r2std = np.zeros(len(sim));
    r3std = np.zeros(len(sim));
    r4std = np.zeros(len(sim))
    r1median = np.zeros(len(sim));
    r2median = np.zeros(len(sim));
    r3median = np.zeros(len(sim));
    r4median = np.zeros(len(sim))
    qerr_r1 = np.zeros(len(sim))
    qerr_r2 = np.zeros(len(sim))
    qerr_r3 = np.zeros(len(sim))
    qerr_r4 = np.zeros(len(sim))

    dft = {}
    df1 = {}
    df2 = {}
    df3 = {}
    df4 = {}

    # df['iB0'] = 0.014

    e = '&'
    d ='$'

    file = open('../Txt/0910_TimeConstant_beta0_' + categorystr + "check.txt", "w")
    file.write(categorystr + '\n')
    file.write(r'\begin{center}')
    file.write(r'\begin{tabular}{  c | c | c | c | c | c | c | c }\\')
    # file.write(r'\begin


    # file.write('\hline' + '\n')
    file.write("Sim " + e + ' R1 ' + e + ' err ' + e +' R2  ' + e + ' err ' + e + '  R3  ' + e +
               ' err ' + e +  "    " + e + ' R4 ' + e + r' err \\' + " "'\n')

    for j in range(len(sim)):
        # print('simarray', sim[j])
        title = str(sim[j]) + '-' + str(team[j])

        r1_down = 2350;
        r1_up = 2400;
        r2_down = 4350;
        r2_up = 4400;
        r3_down = 6350;
        r3_up = 6400;
        r4_down = 8350;
        r4_up = 8400

        if sim[j] == 140:
            r1_down = 2700;
            r1_up = 2740;
            r2_down = 4700;
            r2_up = 4740;
            r3_down = 6700;
            r3_up = 6740;
            r4_down = 8700;
            r4_up = 8740

        dft[j] = df[(df.Sim == sim[j]) & (df.Team == team[j])]
        dft[j].reset_index(inplace=True)

        size = len(dft[j])
        Ums_i = [0] * size
        Ua_i = [0] * size

        Ums_i[0] = dft[j].at[0, 'IM']

        dft[j]['IMminusiB0'] = dft[j]['IM'] - dft[j]['iB0']

        ## only convolute slow part of the signal, which is needed for beta calculation
        for i in range(size - 1):
            # Ua_i = dft[j].at[i+1, 'IMminusiB0']
            Ua_i[i] = dft[j].at[i, 'I_OPM_jma']
            Ua_i[i+1] = dft[j].at[i + 1, 'I_OPM_jma']
            t1 = dft[j].at[i + 1, 'Tsim']
            t2 = dft[j].at[i, 'Tsim']
            Xs = np.exp(-(t1 - t2) / slow)
            Ums_i[i + 1] = Ua_i[i + 1] - (Ua_i[i + 1] - Ums_i[i]) * Xs

        dft[j]['I_conv_slow_jma'] = Ums_i

        df1[j] = dft[j][(dft[j].Tsim >= r1_down) & (dft[j].Tsim < r1_up)]
        df2[j] = dft[j][(dft[j].Tsim >= r2_down) & (dft[j].Tsim < r2_up)]
        df3[j] = dft[j][(dft[j].Tsim >= r3_down) & (dft[j].Tsim < r3_up)]
        df4[j] = dft[j][(dft[j].Tsim >= r4_down) & (dft[j].Tsim < r4_up)]

        if not boolibo:
            r1[j] = np.array((df1[j].IM / (0.10 * df1[j].I_conv_slow_jma)).tolist())
            r2[j] = np.array((df2[j].IM / (0.10 * df2[j].I_conv_slow_jma)).tolist())
            r3[j] = np.array((df3[j].IM / (0.10 * df3[j].I_conv_slow_jma)).tolist())
            r4[j] = np.array((df4[j].IM / (0.10 * df4[j].I_conv_slow_jma)).tolist())

        if boolibo:
            r1[j] = np.array(((df1[j].IM - df1[j].iB0) / (0.10 * df1[j].I_conv_slow_jma)).tolist())
            r2[j] = np.array(((df2[j].IM - df2[j].iB0) / (0.10 * df2[j].I_conv_slow_jma)).tolist())
            r3[j] = np.array(((df3[j].IM - df3[j].iB0) / (0.10 * df3[j].I_conv_slow_jma)).tolist())
            r4[j] = np.array(((df4[j].IM - df4[j].iB0) / (0.10 * df4[j].I_conv_slow_jma)).tolist())
        # print(j, np.nanmean(r1[j]))

        r1mean[j] = np.nanmean(r1[j])
        r1std[j] = np.nanstd(r1[j])
        r2mean[j] = np.nanmean(r2[j])
        r2std[j] = np.nanstd(r2[j])
        r3mean[j] = np.nanmean(r3[j])
        r3std[j] = np.nanstd(r3[j])
        r4mean[j] = np.nanmean(r4[j])
        r4std[j] = np.nanstd(r4[j])

        r1median[j] = np.nanmedian(r1[j])
        r2median[j] = np.nanmedian(r2[j])
        r3median[j] = np.nanmedian(r3[j])
        r4median[j] = np.nanmedian(r4[j])

        qerr_r1[j] = (np.nanquantile(r1[j], 0.75) - np.nanquantile(r1[j], 0.25)) / (2 * 0.6745)
        qerr_r2[j] = (np.nanquantile(r2[j], 0.75) - np.nanquantile(r2[j], 0.25)) / (2 * 0.6745)
        qerr_r3[j] = (np.nanquantile(r3[j], 0.75) - np.nanquantile(r3[j], 0.25)) / (2 * 0.6745)
        qerr_r4[j] = (np.nanquantile(r4[j], 0.75) - np.nanquantile(r4[j], 0.25)) / (2 * 0.6745)

        # lr1 = str(round(r1mean[j], 2)) + ' ' + str(round(r1std[j], 2))
        # lr2 = str(round(r2mean[j], 2)) + ' ' + str(round(r2std[j], 2))
        # lr3 = str(round(r3mean[j], 2)) + ' ' +  str(round(r3std[j], 2))
        # lr4 = str(round(r4mean[j], 2))  + ' ' +  str(round(r4std[j], 2))
        lr5 = str(round(r1median[j], 2)) + ' ' +  str(round(qerr_r1[j], 2))
        lr6 = str(round(r2median[j], 2)) + ' ' +  str(round(qerr_r2[j], 2))
        lr7 = str(round(r3median[j], 2)) + ' ' +  str(round(qerr_r3[j], 2))
        lr8 = str(round(r4median[j], 2)) + ' ' +  str(round(qerr_r4[j], 2))
        lr1 = d + str(round(r1mean[j], 2)) + d + ' ' + str(round(r1std[j], 2))
        lr2 = str(round(r2mean[j], 2)) + ' ' + str(round(r2std[j], 2))
        lr3 = str(round(r3mean[j], 2)) + ' ' +  str(round(r3std[j], 2))
        lr4 = str(round(r4mean[j], 2))  + ' ' +  str(round(r4std[j], 2))

        mat = '$'
        end = ' '

        file.write(title  + e + d + lr1 +d + e + d+ lr2 +d + e + d + lr3 +d  + e + d + lr4 + d + e  + '\n')

    rmean = [r1mean, r2mean, r3mean, r4mean]
    rstd = [r1std, r2std, r3std, r4std]
    rmedian = [r1median, r2median, r3median, r4median]

    qerr_1 = (np.nanquantile(r1median, 0.75) - np.nanquantile(r1median, 0.25)) / (2 * 0.6745)
    qerr_2 = (np.nanquantile(r2median, 0.75) - np.nanquantile(r2median, 0.25)) / (2 * 0.6745)
    qerr_3 = (np.nanquantile(r3median, 0.75) - np.nanquantile(r3median, 0.25)) / (2 * 0.6745)
    qerr_4 = (np.nanquantile(r4median, 0.75) - np.nanquantile(r4median, 0.25)) / (2 * 0.6745)

    qerr = [qerr_1, qerr_2, qerr_3, qerr_4]

    mederr = (np.nanquantile(rmedian[1:4], 0.75) - np.nanquantile(rmedian[1:4], 0.25)) / (2 * 0.6745)

    print('median error', mederr)

    print('qerr', qerr)

    # print('Test mean', len(rmean[0]))
    # print('Test median', len(rmedian[0]))

    file.write("Mean " + ' R1 ' + ' err ' + ' R2  ' + ' err ' +  '  R3  ' + ' err ' + "    " +  ' R4 ' + ' err ' + " "'\n')

    file.write(
        '    ' + end + str(round(np.nanmean(r1mean), 2)) + end + str(round(np.nanstd(r1mean), 2)) + end + str(round(np.nanmean(r2mean), 2)) +
        end + str(round(np.nanstd(r2mean), 2)) + end + str(round(np.nanmean(r3mean), 2)) + end + str(round(np.nanstd(r3mean), 2)) + end
        + str(round(np.nanmean(r4mean), 2)) + end + str(round(np.nanstd(r4mean), 2)) + '\n')

    file.write("Median " + ' R1 ' + ' err ' + ' R2  ' + ' err ' +  '  R3  ' + ' err ' + "    " +  ' R4 ' + ' err ' + " "'\n')

    file.write(
        '   ' + end + str(round(np.nanmedian(r1median), 2)) + end + str(round(qerr_1, 2)) +
        end + str(round(np.nanmedian(r2median), 2)) + end + str(round(qerr_2, 2)) +
        end + str(round(np.nanmedian(r3median), 2)) + end + str(round(qerr_3, 2)) +
        end + str(round(np.nanmedian(r4median), 2)) + end + str(round(qerr_4, 2)) + '\n')


    file.write('Mean R2-R4 &' + end + str(round(np.nanmean(rmean[1:4]), 2)) + end + str(
        round(np.nanstd(rmean[1:4]), 2)) + end + 'Mean R1-R4 &' + end + str(
        round(np.nanmean(rmean), 2)) + end + str(
        round(np.nanstd(rmean), 2)) + end + '\n')
    file.write('Median R2-R4 &' + end + str(round(np.nanmedian(rmedian[1:4]), 2)) + end + str(
        round(mederr, 2)) + end +  'Median R1-R4 &' + end + str(round(np.nanmedian(rmedian), 2)) + end + str(
        round(np.nanstd(mederr), 2)) + end  + '\n')

    return rmean, rstd, rmedian, qerr



def ratiofunction_beta_9602_xls(df, sim, team, categorystr, boolib0, slow, fast):
    r1 = [0] * len(sim);
    r2 = [0] * len(sim);

    r1mean = np.zeros(len(sim));
    r2mean = np.zeros(len(sim));
    r1std = np.zeros(len(sim));
    r2std = np.zeros(len(sim));

    r1median = np.zeros(len(sim))
    r2median = np.zeros(len(sim))
    qerr_r1 = np.zeros(len(sim))
    qerr_r2 = np.zeros(len(sim))

    dft = {}
    df1 = {}
    df2 = {}

    df['iB0'] = 0.014

    Pval_komhyr = np.array([1000, 199, 59, 30, 20, 10, 7, 5])
    komhyr_sp_tmp = np.array([1, 1.007, 1.018, 1.022, 1.032, 1.055, 1.070, 1.092])
    komhyr_en_tmp = np.array([1, 1.007, 1.018, 1.029, 1.041, 1.066, 1.087, 1.124])

    komhyr_sp = [1 / i for i in komhyr_sp_tmp]
    komhyr_en = [1 / i for i in komhyr_en_tmp]

    file = open('../Txt/9602_TimeConstant_' + categorystr + "5secslatex_table.txt", "w")
    file.write(categorystr + '\n')
    file.write(" " + ' mean ' + '   ' + ' mean  ' + '   ' +  '  median  ' + '   ' + "    " +  ' median ' + '  ' + " "'\n')

    file.write("Sim " + ' R1 ' + ' err ' + ' R2  ' + ' err ' +  '  R1  ' + ' err ' + "    " +  ' R2 ' + ' err ' + " "'\n')


    print('len sim', len(sim))

    for j in range(len(sim)):
        # print('simarray', sim[j])
        title = str(sim[j]) + '-' + str(team[j])

        dftj = df[(df.Sim == sim[j]) & (df.Team == team[j])]
        dftj.reset_index(inplace=True)

        size = len(dftj)
        Ums_i = [0] * size
        Ua_i = [0] * size
        Ums_i[0] = dftj.at[0, 'IM']

        ## only convolute slow part of the signal, which is needed for beta calculation
        for k in range(size - 1):

            ## komhyr corrections
            for p in range(len(komhyr_en) - 1):

                if dftj.at[k, 'ENSCI'] == 1:
                    if (dftj.at[k, 'Pair'] >= Pval_komhyr[p + 1]) & (dftj.at[k, 'Pair'] < Pval_komhyr[p]):
                        # print(p, Pval[p + 1], Pval[p ])
                        dftj.at[k, 'I_OPM_komhyr'] = dftj.at[k, 'PO3_OPM'] * dftj.at[k, 'PFcor'] * komhyr_en[p] / \
                                                     (dftj.at[k, 'TPint'] * 0.043085)
                if dftj.at[k, 'ENSCI'] == 0:
                    if (dftj.at[k, 'Pair'] >= Pval_komhyr[p + 1]) & (dftj.at[k, 'Pair'] < Pval_komhyr[p]):
                        # print(p, Pval[p + 1], Pval[p ])
                        dftj.at[k, 'I_OPM_komhyr'] = dftj.at[k, 'PO3_OPM'] * dftj.at[k, 'PFcor'] * komhyr_sp[p] / \
                                                     (dftj.at[k, 'TPint'] * 0.043085)

            if (dftj.at[k, 'Pair'] <= Pval_komhyr[7]):

                if dftj.at[k, 'ENSCI'] == 1:
                    dftj.at[k, 'I_OPM_komhyr'] = dftj.at[k, 'PO3_OPM'] * dftj.at[k, 'PFcor'] * komhyr_en[7] / \
                                                 (dftj.at[k, 'TPint'] * 0.043085)
                if dftj.at[k, 'ENSCI'] == 0:
                    dftj.at[k, 'I_OPM_komhyr'] = dftj.at[k, 'PO3_OPM'] * dftj.at[k, 'PFcor'] * komhyr_sp[7] / \
                                                 (dftj.at[k, 'TPint'] * 0.043085)

            size = len(dftj)
            Ums_i = [0] * size
            Ua_i = [0] * size
            Ums_i[0] = dftj.at[0, 'IM']

            ## only convolute slow part of the signal, which is needed for beta calculation
        for ik in range(size - 1):
            Ua_i = dftj.at[ik + 1, 'I_OPM_jma']
            t1 = dftj.at[ik + 1, 'Tsim']
            t2 = dftj.at[ik, 'Tsim']
            Xs = np.exp(-(t1 - t2) / slow)
            Ums_i[ik + 1] = Ua_i - (Ua_i - Ums_i[ik]) * Xs


        dftj['I_conv_slow_jma'] = Ums_i

        year = (dftj.iloc[0]['JOSIE_Nr'])

        rt1 = (dftj.iloc[0]['R1_Tstop'])
        rt2 = (dftj.iloc[0]['R2_Tstop'])
        if sim[j] == 90: rt1 = 1050
        t1 = (dftj.Tsim <= rt1) & (dftj.Tsim >= rt1 - 5)
        t2 = (dftj.Tsim <= rt2) & (dftj.Tsim >= rt2 - 5)
        #
        df1[j] = dftj[t1]
        df2[j] = dftj[t2]
        # c = df1[j].IM .tolist()
        # sc = ( df1[j].I_conv_slow.tolist())

        if not boolib0:
            r1[j] = np.array((df1[j].IM / (0.10 * df1[j].I_conv_slow_jma)).tolist())
            r2[j] = np.array((df2[j].IM / (0.10 * df2[j].I_conv_slow_jma)).tolist())

        if boolib0:
            r1[j] = np.array(((df1[j].IM - df1[j].iB0) / (0.10 * df1[j].I_conv_slow_jma)).tolist())
            r2[j] = np.array(((df2[j].IM - df2[j].iB0) / (0.10 * df2[j].I_conv_slow_jma)).tolist())

        # print(sim[j],team[j], 'Ratio', r1[j], r2[j])
        # print(sim[j], team[j], 'Curent', df1[j].IM.tolist(), df1[j].I_conv_slow.tolist() )
        #
        r1mean[j] = np.nanmean(r1[j])
        r2mean[j] = np.nanmean(r2[j])
        r1std[j] = np.nanstd(r1[j])
        r2std[j] = np.nanstd(r2[j])
        r1median[j] = np.nanmedian(r1[j])
        r2median[j] = np.nanmedian(r2[j])
        qerr_r1[j] = (np.nanquantile(r1[j], 0.75) - np.nanquantile(r1[j], 0.25)) / (2 * 0.6745)
        qerr_r2[j] = (np.nanquantile(r2[j], 0.75) - np.nanquantile(r2[j], 0.25)) / (2 * 0.6745)

        end = ' '
        lr1 = str(round(r1mean[j], 2)) + end + str(round(r1std[j], 2))
        lr2 = str(round(r2mean[j], 2)) + end + str(round(r2std[j], 2))
        lr3 = str(round(r1median[j], 2)) + end + str(round(qerr_r1[j], 2))
        lr4 = str(round(r2median[j], 2)) + end + str(round(qerr_r2[j], 2))


        file.write(str(int(year)) + '-' + title + end + lr1 + end + lr2 + end + lr3 + end + lr4 + '\n')

    rmean = [r1mean, r2mean]
    rstd = [r1std, r2std]
    rmedian = [r1median, r2median]
    rqerr = [qerr_r1, qerr_r2]

    qerr_1 = (np.nanquantile(r1median, 0.75) - np.nanquantile(r1median, 0.25)) / (2 * 0.6745)
    qerr_2 = (np.nanquantile(r2median, 0.75) - np.nanquantile(r2median, 0.25)) / (2 * 0.6745)

    mederr = (np.nanquantile(rmedian, 0.75) - np.nanquantile(rmedian, 0.25)) / (2 * 0.6745)


    file.write(
        'Mean  ' + end + str(round(np.nanmean(r1mean), 2)) + end + str(round(np.nanstd(r1mean), 2)) + end + str(round(np.nanmean(r2mean), 2))
        + end + str(round(np.nanstd(r2mean), 2)) +  '\n')
    file.write(
        'Median   ' + end + str(round(np.nanmedian(r1median), 2)) + end + str(round(qerr_1, 2)) + end + str(round(np.nanmedian(r2median), 2)) +
        end + str(round(qerr_2, 2)) + end + r'\\' + '\n')

    file.write('Mean R1-R2 ' + end + str(round(np.nanmean(rmean), 2)) + end + str(
        round(np.nanstd(rmean), 2)) + '\n')
    file.write('Median R1-R2 &' + end + str(round(np.nanmedian(rmedian), 2)) + end + str(
        round(mederr, 2)) +  '\n')

    file.close()

    return rmean, rstd, rmedian, rqerr

def ratiofunction_beta_oneforonesim(df, sim, team, categorystr, boolibo, slow):
    ##modifications made for timzescan, i slow conv used for beta calculation will be calculated in this function!

    r1 = [0] * len(sim);
    r2 = [0] * len(sim);
    r3 = [0] * len(sim);
    r4 = [0] * len(sim)

    r1mean = np.zeros(len(sim));
    r2mean = np.zeros(len(sim));
    r3mean = np.zeros(len(sim));
    r4mean = np.zeros(len(sim))
    r1std = np.zeros(len(sim));
    r2std = np.zeros(len(sim));
    r3std = np.zeros(len(sim));
    r4std = np.zeros(len(sim))
    r1median = np.zeros(len(sim));
    r2median = np.zeros(len(sim));
    r3median = np.zeros(len(sim));
    r4median = np.zeros(len(sim))
    qerr_r1 = np.zeros(len(sim))
    qerr_r2 = np.zeros(len(sim))
    qerr_r3 = np.zeros(len(sim))
    qerr_r4 = np.zeros(len(sim))
    beta = np.zeros(len(sim))

    dft = {}
    df1 = {}
    df2 = {}
    df3 = {}
    df4 = {}

    dftab = pd.DataFrame()


    for j in range(len(sim)):
        # print('simarray', sim[j])
        title = str(sim[j]) + '-' + str(team[j])

        r1_down = 2350 -20 ;
        r1_up = 2400 -20;
        r2_down = 4350 -20;
        r2_up = 4400-20;
        r3_down = 6350-20;
        r3_up = 6400-20;
        r4_down = 8350 -20;
        r4_up = 8400 -20

        if sim[j] == 140:
            r1_down = 2700;
            r1_up = 2740;
            r2_down = 4700;
            r2_up = 4740;
            r3_down = 6700;
            r3_up = 6740;
            r4_down = 8700;
            r4_up = 8740

        if sim[j] == 166:
            r1_down = 2350 + 180;
            r1_up = 2400 + 180;
            r2_down = 4350 + 180;
            r2_up = 4400 + 180;
            r3_down = 6350 + 180;
            r3_up = 6400 + 180;
            r4_down = 8350 + 180;
            r4_up = 8400 + 180

        if sim[j] == 161:
            r1_down = 2350 + 30;
            r1_up = 2400 + 30;
            r2_down = 4350+ 30;
            r2_up = 4400 + 30;
            r3_down = 6350 + 30;
            r3_up = 6400 + 30;
            r4_down = 8350;
            r4_up = 8400

        if sim[j] == 162:
            r1_down = 2350 +30;
            r1_up = 2400 +30;
            r2_down = 4350+30;
            r2_up = 4400 +30;
            r3_down = 6350 +30;
            r3_up = 6400 +30;
            r4_down = 8350 +30;
            r4_up = 8400 +30

        if (sim[j] == 143):
            # | (sim[j] == 161)
            r4_down = 8350 + 60;
            r4_up = 8400 + 60
        # if (sim[j] == 137) | (sim[j] == 141) | (sim[j] == 149):

        # if sim[j] != 166: continue

        dft[j] = df[(df.Sim == sim[j]) & (df.Team == team[j])]
        dft[j].reset_index(inplace=True)

        size = len(dft[j])
        Ums_i = [0] * size
        Ua_i = [0] * size

        Ums_i[0] = dft[j].at[0, 'IM']

        # dft[j]['IMminusiB0'] = dft[j]['IM'] - dft[j]['iB0']

        ## only convolute slow part of the signal, which is needed for beta calculation
        for i in range(size - 1):
            # Ua_i = dft[j].at[i+1, 'IMminusiB0']
            Ua_i[i] = dft[j].at[i, 'I_OPM_jma']
            Ua_i[i+1] = dft[j].at[i + 1, 'I_OPM_jma']
            t1 = dft[j].at[i + 1, 'Tsim']
            t2 = dft[j].at[i, 'Tsim']
            Xs = np.exp(-(t1 - t2) / slow)
            Ums_i[i + 1] = Ua_i[i + 1] - (Ua_i[i + 1] - Ums_i[i]) * Xs
        fi = dft[j].first_valid_index()
        li = dft[j].last_valid_index()
        dft[j].loc[:, 'I_conv_slow_jma'] = Ums_i
        # dft[j]['I_conv_slow_jma'] = Ums_i


        df1[j] = dft[j][(dft[j].Tsim >= r1_down) & (dft[j].Tsim < r1_up)]
        df2[j] = dft[j][(dft[j].Tsim >= r2_down) & (dft[j].Tsim < r2_up)]
        df3[j] = dft[j][(dft[j].Tsim >= r3_down) & (dft[j].Tsim < r3_up)]
        df4[j] = dft[j][(dft[j].Tsim >= r4_down) & (dft[j].Tsim < r4_up)]

        if not boolibo:
            r1[j] = np.array((df1[j].IM / (df1[j].I_conv_slow_jma)).tolist())
            r2[j] = np.array((df2[j].IM / (df2[j].I_conv_slow_jma)).tolist())
            r3[j] = np.array((df3[j].IM / (df3[j].I_conv_slow_jma)).tolist())
            r4[j] = np.array((df4[j].IM / (df4[j].I_conv_slow_jma)).tolist())

        if boolibo:
            r1[j] = np.array(((df1[j].IM - df1[j].iB0) / (df1[j].I_conv_slow_jma)).tolist())
            r2[j] = np.array(((df2[j].IM - df2[j].iB0) / (df2[j].I_conv_slow_jma)).tolist())
            r3[j] = np.array(((df3[j].IM - df3[j].iB0) / (df3[j].I_conv_slow_jma)).tolist())
            r4[j] = np.array(((df4[j].IM - df4[j].iB0) / (df4[j].I_conv_slow_jma)).tolist())
        # print(j, np.nanmean(r1[j]))

        r1median[j] = np.nanmedian(r1[j])
        r2median[j] = np.nanmedian(r2[j])
        r3median[j] = np.nanmedian(r3[j])
        r4median[j] = np.nanmedian(r4[j])

        # print('medians in the function', r1median[j], r2median[j], r3median[j], r4median[j])

        dftab.loc[j,'Sim'] = title
        dftab.loc[j,'iB0'] = dft[j].loc[dft[j].first_valid_index(),'iB0']
        dftab.loc[j,'iB1'] = dft[j].loc[dft[j].first_valid_index(),'iB1']
        dftab.loc[j,'iB1-iB0'] = dft[j].loc[dft[j].first_valid_index(),'iB1'] - dft[j].loc[dft[j].first_valid_index(),'iB0']

        # print(j, 'r1median[j]', r1median[j])

        dftab.loc[j,'beta1'] = r1median[j]
        dftab.loc[j,'beta2'] = r2median[j]
        dftab.loc[j,'beta3'] = r3median[j]
        dftab.loc[j,'beta4'] = r4median[j]
        beta[j] = np.nanmedian[r1median[j], r2median[j], r3median[j], r4median[j]]
        dftab.loc[j,'beta'] = beta[j]

        # print(j, title, 'rmedians',r1median[j], r2median[j], r3median[j], r4median[j])

        df.loc[(df.Sim == sim[j]) & (df.Team == team[j]), 'beta'] = beta[j]

        err = [0] * len(list(dftab))
        err2 = [0] * len(list(dftab))

        for i, k in zip(list(dftab), range(len(list(dftab)))):
            # err[k] = (np.nanquantile(dftab[i].tolist(), 0.75) - np.nanquantile(dftab[i].tolist(), 0.25)) / (2 * 0.6745)
            # print('tab loop', i, k)
            if k == 0: continue
            err2[k] = stats.median_abs_deviation(dftab[i].tolist())

        # plt.close('all')
        # # if sim[j] == 144:
        # fig, axs = plt.subplots(figsize=(8, 6))
        # plt.plot(np.array(dft[j]['Tsim'] / 60), np.array(dft[j]['I_OPM_jma']), label='I OPM.')
        # plt.plot(np.array(dft[j]['Tsim'] / 60), np.array(dft[j]['I_conv_slow_jma']), label='I slow conv.')
        # plt.plot(np.array(dft[j]['Tsim'] / 60), np.array(dft[j]['IM']), label='I ECC')
        # plt.plot(np.array(dft[j]['Tsim'] / 60), np.array(dft[j]['iB0']), label='iB0')
        # plt.plot(np.array(dft[j]['Tsim'] / 60), np.array(dft[j]['iB1']), label='iB1')
        # plt.plot(np.array(dft[j]['Tsim'] / 60), np.array(dft[j]['iB1'] - dft[j]['iB0']), label='iB1 - iB0')
        # # plt.plot(np.array(dft[j]['time']/60), np.array(dft[j]['I_gen_pr']), label ='I gen. previous')
        # plt.axvline(x=r1_down / 60, color='red', linestyle='--', label='t begin [1]', linewidth=1)
        # plt.axvline(x=r1_up / 60, color='black', linestyle='--', label='t end [1]', linewidth=1)
        # plt.axvline(x=r2_down / 60, color='red', linestyle='--', linewidth=1)
        # plt.axvline(x=r2_up / 60, color='black', linestyle='--', linewidth=1)
        # plt.axvline(x=r3_down / 60, color='red', linestyle='--', linewidth=1)
        # plt.axvline(x=r3_up / 60, color='black', linestyle='--', linewidth=1)
        # plt.axvline(x=r4_down / 60, color='red', linestyle='--', linewidth=1)
        # plt.axvline(x=r4_up / 60, color='black', linestyle='--', linewidth=1)
        # # plt.ylim([-0.5, 10])
        # axs.legend(loc='upper right', frameon=True, fontsize='small')
        # axs.set_yscale('log')
        # # plotname = sondenumber + ' ' + str(exp)
        # plt.title(categorystr + ' ' + title)
        # #
        # path = '/home/poyraden/Analysis/JosieAnalysis/Plots/Beta_0910_Plots/'
        # # plt.savefig(path + categorystr + '_' + title + '_v2.eps')
        # plt.savefig(path + categorystr + '_' + title + '_log.png')
        # # plt.show()
        # #v2 is every gap is 10 seconds earlier
        # plt.close()

        r1mean[j] = np.nanmean(r1[j])
        r1std[j] = np.nanstd(r1[j])
        r2mean[j] = np.nanmean(r2[j])
        r2std[j] = np.nanstd(r2[j])
        r3mean[j] = np.nanmean(r3[j])
        r3std[j] = np.nanstd(r3[j])
        r4mean[j] = np.nanmean(r4[j])
        r4std[j] = np.nanstd(r4[j])

        err2[k] = stats.median_abs_deviation(dftab[i].tolist())

        qerr_r1[j] = stats.median_abs_deviation(r1[j])
        qerr_r2[j] = stats.median_abs_deviation(r2[j])
        qerr_r3[j] = stats.median_abs_deviation(r3[j])
        qerr_r4[j] = stats.median_abs_deviation(r4[j])

    rmean = [r1mean, r2mean, r3mean, r4mean]
    rstd = [r1std, r2std, r3std, r4std]
    # rmedianarray = [r1median, r2median, r3median, r4median]

    rmed1 = np.nanmedian(r1median)
    rmed2 = np.nanmedian(r2median)
    rmed3 = np.nanmedian(r3median)
    rmed4 = np.nanmedian(r4median)

    rmedian = [rmed1, rmed2, rmed3, rmed4]


    # print('rmedian funtion', rmedian)

    qerr_1 = stats.median_abs_deviation(r1median)
    qerr_2 = stats.median_abs_deviation(r2median)
    qerr_3 = stats.median_abs_deviation(r3median)
    qerr_4 = stats.median_abs_deviation(r4median)

    qerr = [qerr_1, qerr_2, qerr_3, qerr_4]


    pathtab = '/home/poyraden/Analysis/JosieAnalysis/csv/0910_'
    dftab.to_csv(pathtab + categorystr + 'beta_cor_upd.csv')

    dftab.loc["Mean"] = dftab.mean()
    dftab.loc["std"] = dftab.std()
    dftab.loc["Median"] = dftab.median()
    dftab.loc['median error'] = err2

    dftab.loc['beta1-4 median'] = np.median(rmedian)
    # print((r1median,rmedian[2],rmedian[3]))
    m14 = np.hstack((r1median, r2median,r3median,r4median)).ravel()
    m24 = np.hstack((r2median,r3median,r4median)).ravel()
    dftab.loc['beta1-4 median error'] = stats.median_abs_deviation(m14)
    dftab.loc['beta2-4 median'] = np.median(rmedian[1:4])
    dftab.loc['beta2-4 median error'] = stats.median_abs_deviation(m24)
    dftab = dftab.round(3)

    # print(dftab)

    pathtab = '/home/poyraden/Analysis/JosieAnalysis/Codes/tables/'
    dftab.to_latex(pathtab + categorystr + 'beta_cor_upd.tex', index = False)
    dftab.to_excel(pathtab + categorystr + 'beta_cor_upd.xlsx')
    # print('end of', categorystr)

    beta_all = np.median(beta)
    beta_err = stats.median_abs_deviation(beta)

    return rmean, rstd, rmedian, qerr, df, beta_all, beta_err
