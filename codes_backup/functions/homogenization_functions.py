import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from re import search

pval = np.array([1100, 200, 100, 50, 30, 20, 10, 7, 5, 3])
pvallog = [np.log10(i) for i in pval]

komhyr_86 = np.array([1, 1, 1.007, 1.018, 1.022, 1.032, 1.055, 1.070, 1.092, 1.124])  # SP Komhyr
komhyr_95 = np.array([1, 1, 1.007, 1.018, 1.029, 1.041, 1.066, 1.087, 1.124, 1.241])  # ENSCI Komhyr

komhyr_86_unc = np.array([0, 0, 0.005, 0.006, 0.008, 0.009, 0.010, 0.012, 0.014, 0.025])  # SP Komhyr
komhyr_95_unc = np.array([0, 0, 0.005, 0.005, 0.008, 0.012, 0.023, 0.024, 0.024, 0.043])  # ECC Komhyr


def return_phipcor(df, phip_grd, unc_phip_grd, cpf, unc_cpf):
    # O3S-DQA 8.5

    df['Phip_cor'] = df[phip_grd] / df[cpf]  # Eq. 22
    df['unc_Phip_cor'] = df['Phip_cor'] * np.sqrt(
        df[unc_phip_grd] ** 2 / df[phip_grd] ** 2 + df[unc_cpf] ** 2 / df[cpf] ** 2)  # Eq. 23

    return df['Phip_cor'], df['unc_Phip_cor']

def pumptemp_corr(df, boxlocation, temp, unc_temp, pair):
    '''
    O3S-DQA 8.3
    :param df: dataframe
    :param boxlocation: location of the temperature measurement
    :param temp: temp. of the pump that was measured at boxlocation
    :param pressure: pressure of the air
    :return: the corrected base temperature of the pump
    '''
    df['Tpump_cor'] = 0
    df['unc_Tpump_cor'] = 0

    if (boxlocation == 'Box') | (boxlocation == 'case1'):  # case I in O3S-DQA guide
        df.loc[(df[pair] >= 40), 'deltat'] = 7.43 - 0.393 * np.log10(df.loc[(df[pair] >= 40), pair])
        df.loc[(df[pair] < 40) & (df[pair] > 6), 'deltat'] = 2.7 + 2.6 * np.log10(
            df.loc[(df[pair] < 40) & (df[pair] > 6), pair])
        df.loc[(df[pair] <= 6), 'deltat'] = 4.5
        df['unc_deltat'] = 1  # units in K

    if (boxlocation == 'ExternalPumpTaped') | (boxlocation == 'case2') | (
            boxlocation == 'case3'):  # case III in O3S-DQA guide
        df.loc[(df[pair] > 70), 'deltat'] = 20.6 - 6.7 * np.log10(df.loc[(df[pair] > 70), pair])
        df.loc[(df[pair] > 70), 'unc_deltat'] = 3.9 - 1.13 * np.log10(df.loc[(df[pair] > 70), pair])
        df.loc[(df[pair] <= 70) & (df[pair] >= 15), 'deltat'] = 8.25
        # updated formula, 17/06/2021 not 3.25 - 4.25 ... but 3.25 + 4.25
        # df.loc[(df[pair] < 15) & (df[pair] >= 5), 'deltat'] = 3.25 + 4.25 * np.log10(
        #     df.loc[(df[pair] < 15) & (df[pair] >= 5), pair])
        # updated in 03/03/2022 to have pimn 5 to 3
        df.loc[(df[pair] < 15) & (df[pair] >= 3), 'deltat'] = 3.25 + 4.25 * np.log10(
            df.loc[(df[pair] < 15) & (df[pair] >= 3), pair])
        df.loc[(df[pair] <= 70), 'unc_deltat'] = 0.3 + 1.13 * np.log10(df.loc[(df[pair] <= 70), pair])

    if (boxlocation == 'ExternalPumpGlued') | (boxlocation == 'case4'):  # case IV in O3S-DQA guide
        df.loc[(df[pair] > 40), 'deltat'] = 6.4 - 2.14 * np.log10(df.loc[(df[pair] > 40), pair])
        df.loc[(df[pair] <= 40) & (df[pair] >= 3), 'deltat'] = 3.0
        df['unc_deltat'] = 0.5  # units in K

    filt = df[pair] > 3

    if (boxlocation == 'InternalPump') | (boxlocation == 'case5'):  # case V in O3S-DQA guide
        df.loc[filt, 'deltat'] = 0  # units in K
        df.loc[filt, 'unc_deltat'] = 0  # units in K
    df.loc[(df[pair] > 3), 'deltat_ppi'] = 3.9 - 0.8 * np.log10(df.loc[(df[pair] > 3), pair])  # Eq. 12
    df.loc[(df[pair] > 3), 'unc_deltat_ppi'] = 0.5

    df.loc[filt, 'Tpump_cor'] = df.loc[filt, temp] + df.loc[filt, 'deltat'] + df.loc[filt, 'deltat_ppi']  # Eq. 13
    df.loc[filt, 'unc_Tpump_cor'] = (df.loc[filt, unc_temp] ** 2 / df.loc[filt, temp] ** 2) + \
                                    (df.loc[filt, 'unc_deltat'] ** 2 / df.loc[filt, temp] ** 2) + (
                                                df.loc[filt, 'unc_deltat_ppi'] ** 2 / df.loc[filt, temp] ** 2)  # Eq. 14


    return df.loc[filt, 'Tpump_cor'], df.loc[filt, 'unc_Tpump_cor']

def VecInterpolate_log(XValues, YValues, unc_YValues, dft, Pair):
    # dft = dft.reset_index()
    dft['Cpf'] = 1
    dft['unc_Cpf'] = 1
    dft['plog'] = np.log10(dft[Pair])

    for k in range(len(dft)):
        dft.at[k, 'Cpf'] = 1

        for i in range(len(XValues) - 1):
            # check that value is in between xvalues
            if (XValues[i] >= dft.at[k, 'plog'] >= XValues[i + 1]):
                x1 = float(XValues[i])
                x2 = float(XValues[i + 1])
                y1 = float(YValues[i])
                y2 = float(YValues[i + 1])
                unc_y1 = float(unc_YValues[i])
                unc_y2 = float(unc_YValues[i + 1])
                dft.at[k, 'Cpf'] = y1 + (dft.at[k, 'plog'] - x1) * (y2 - y1) / (x2 - x1)
                # if k > 500:
                # print('Cpf in function',k,  dft.loc[k,'Cpf'])
                dft.at[k, 'unc_Cpf'] = unc_y1 + (dft.at[k, 'plog'] - x1) * (unc_y2 - unc_y1) / (x2 - x1)

    return dft['Cpf'], dft['unc_Cpf']
    # return dft

def VecInterpolate_linear(XValues, YValues, unc_YValues, dft, Pair):
    dft = dft.reset_index()

    for k in range(len(dft)):

        for i in range(len(XValues) - 1):
            # check that value is in between xvalues
            if (XValues[i] >= dft.at[k, Pair] >= XValues[i + 1]):
                x1 = float(XValues[i])
                x2 = float(XValues[i + 1])
                y1 = float(YValues[i])
                y2 = float(YValues[i + 1])
                unc_y1 = float(unc_YValues[i])
                unc_y2 = float(unc_YValues[i + 1])
                dft.at[k, 'Cpf'] = y1 + (dft.at[k, Pair] - x1) * (y2 - y1) / (x2 - x1)
                dft.at[k, 'unc_Cpf'] = unc_y1 + (dft.at[k, Pair] - x1) * (unc_y2 - unc_y1) / (x2 - x1)

    return dft['Cpf'], dft['unc_Cpf']

def pumpflow_efficiency(df, pair, pumpcorrectiontag, effmethod):
    '''
    O3S-DQA 8.5 based on Table 6
    :param df:
    :param pair:
    :param pumpcorrectiontag:
    :param effmethod:
    :return:
    '''

    df['Cpf'] = 1
    df['unc_Cpf'] = 1

    if effmethod == 'polyfit':

        if pumpcorrectiontag == 'komhyr_95':
            df['Cpf'] = 2.17322861 - 3.686021555 * np.log10(df[pair]) + 5.105113826 * (
                np.log10(df[pair])) ** 2 - 3.741595297 * (np.log10(df[pair])) ** 3 + 1.496863681 * (
                            np.log10(df[pair])) ** 4 - \
                        0.3086952232 * (np.log10(df[pair])) ** 5 + 0.02569158956 * (np.log10(df[pair])) ** 6
            df['unc_Cpf'] = 0.07403603165 - 0.08532895578 * np.log10(df[pair]) + 0.03463984997 * (
                np.log10(df[pair])) ** 2 - 0.00462582698 * (np.log10(df[pair])) ** 3

    if effmethod == 'table_interpolate':

        if pumpcorrectiontag == 'komhyr_86':
            # df['Cpf'], df['unc_Cpf'] = VecInterpolate_linear(pval, komhyr_86, komhyr_86_unc,  df, pair)
            df['Cpf'], df['unc_Cpf'] = VecInterpolate_log(pvallog, komhyr_86, komhyr_86_unc, df, pair)
            # df = VecInterpolate_log(pvallog, komhyr_86, komhyr_86_unc,  df, pair)

        if pumpcorrectiontag == 'komhyr_95':
            # df['Cpf'], df['unc_Cpf'] = VecInterpolate_linear(pval, komhyr_95, komhyr_95_unc,  df, pair)
            df['Cpf'], df['unc_Cpf'] = VecInterpolate_log(pvallog, komhyr_95, komhyr_95_unc, df, pair)


    if effmethod == 'table_interpolate_nolog':

        if pumpcorrectiontag == 'komhyr_86':
            df['Cpf'], df['unc_Cpf'] = VecInterpolate_linear(pval, komhyr_86, komhyr_86_unc,  df, pair)
            # df['Cpf'], df['unc_Cpf'] = VecInterpolate_log(pvallog, komhyr_86, komhyr_86_unc, df, pair)
            # df = VecInterpolate_log(pvallog, komhyr_86, komhyr_86_unc,  df, pair)

        if pumpcorrectiontag == 'komhyr_95':
            df['Cpf'], df['unc_Cpf'] = VecInterpolate_linear(pval, komhyr_95, komhyr_95_unc,  df, pair)
            # df['Cpf'], df['unc_Cpf'] = VecInterpolate_log(pvallog, komhyr_95, komhyr_95_unc, df, pair)


    # return df
    return df['Cpf'], df['unc_Cpf']
