import pandas as pd
import numpy as np

from constant_variables import *


def smooth_gaussian(dff, vtime, vozcur, twindow, sigma):

    time = np.array(dff[vtime])
    ozcur = np.array(dff[vozcur])
    n = len(dff)
    smooth = np.zeros(n)

    for ir in range(n):
        timeir = time[ir]
        ir1 = np.searchsorted(time, timeir - twindow, side='left')
        ir2 = np.searchsorted(time, timeir + twindow, side='right')
        expcoeffsum = np.sum(np.exp(-((time[ir1:ir2+1] - timeir)**2) / (2 * (sigma ** 2))) * ozcur[ir1:ir2+1])
        factor = np.sum(np.exp(-((time[ir1:ir2+1] - timeir)**2) / (2 * (sigma ** 2))))
        smooth[ir] = expcoeffsum / factor
    dff['tmp'] = smooth

    return dff['tmp']

def calibrate_rdif(dft, bool_9602, bool_0910, bool_2017):
    
    #en0505
    # 0[9.08348057 - 2.21309635]
    #en1010
    # 1[11.16603729 - 2.10008198]
    #sp0505
    # 2[4.85429838 - 2.26439631]
    # sp 1010
    # 3[7.2149126 - 1.95924339]

    filtEN = dft.ENSCI == 1
    filtSP = dft.ENSCI == 0

    filtS10 = dft.Sol == 1
    filtS05 = dft.Sol == 0.5

    filtB10 = dft.Buf == 1.0
    filtB05 = dft.Buf == 0.5
    filtB01 = dft.Buf == 0.1

    filterEN0505 = (filtEN & filtS05 & filtB05)
    filterEN1010 = (filtEN & filtS10 & filtB10)
    filterEN1001 = (filtEN & filtS10 & filtB01)

    profEN0505 = dft.loc[filterEN0505]
    profEN1010 = dft.loc[filterEN1010]
    # 2017
    if bool_2017:
        profEN1010 = dft.loc[filterEN1001]

    profEN0505['R'] = 9.08 - 2.21 * np.log10(profEN0505['Pair'])
    profEN0505['I_Calib'] = profEN0505['Ifast_minib0_deconv_sm10'] * (100 / (100 + profEN0505['R']))
    profEN1010['R'] = 11.16 - 2.10 * np.log10(profEN1010['Pair'])
    profEN1010['I_Calib'] = profEN1010['Ifast_minib0_deconv_sm10'] * (100 / (100 + profEN1010['R']))

    profEN0505_nodup = profEN0505.drop_duplicates(['Sim', 'Team'])
    profEN1010_nodup = profEN1010.drop_duplicates(['Sim', 'Team'])

    print(profEN0505_nodup[['Sim', 'Team']])

    if not bool_9602:
        totO3_EN0505 = profEN0505_nodup.frac.mean()
        totO3_EN1010 = profEN1010_nodup.frac.mean()
    if bool_9602:
        totO3_EN0505 = 1
        totO3_EN1010 = 1

    filterSP1010 = (filtSP & filtS10 & filtB10)
    filterSP0505 = (filtSP & filtS05 & filtB05)
    filterSP1001 = (filtSP & filtS10 & filtB01)

    profSP1010 = dft.loc[filterSP1010]
    profSP0505 = dft.loc[filterSP0505]
    # 2017
    if bool_2017:
        profSP0505 = dft.loc[filterSP1001]

    profSP0505['R'] = 4.85 - 2.26 * np.log10(profSP0505['Pair'])
    profSP0505['I_Calib'] = profSP0505['Ifast_minib0_deconv_sm10'] * (100 / (100 + profSP0505['R']))
    profSP1010['R'] = 7.21 - 1.95 * np.log10(profSP1010['Pair'])
    profSP1010['I_Calib'] = profSP1010['Ifast_minib0_deconv_sm10'] * (100 / (100 + profSP1010['R']))

    profSP1010_nodup = profSP1010.drop_duplicates(['Sim', 'Team'])
    profSP0505_nodup = profSP0505.drop_duplicates(['Sim', 'Team'])

    if not bool_9602:
        totO3_SP1010 = profSP1010_nodup.frac.mean()
        totO3_SP0505 = profSP0505_nodup.frac.mean()
    if bool_9602:
        totO3_SP1010 = 1
        totO3_SP0505 = 1

    prof = [profEN0505, profEN1010, profSP0505, profSP1010]

    return prof

def smooth_parabolic(dft, current, time, tau):
    n1 = 0  # no need to initialize n1, you can loop over syarting from 0
    n2 = len(dft)
    n = tau + 1
    smooth = [0.] * n2

    for ir in range(n1, n2 - 1):
        timeir = dft.at[ir, 'Tsim']
        id1 = np.argmin(abs(time - (timeir - n)))
        id2 = np.argmin(abs(time - (timeir + n)))
        ir1 = max(id1, n1)
        ir2 = min(id2, n2 - 1)
        #         print('ir1: ', ir1, 'ir2:' , ir2)
        c0 = 0.
        c1 = 0.
        i = ir1
        while i <= ir2:
            f = 1. - ((float(time[i]) - float(timeir)) / tau) ** 2
            if f <= 0:
                i = i + 1
            else:
                c0 = c0 + f
                #                 ozcur[i] = dft.loc[i, 'Ifast_minib0_deconv']
                c1 = c1 + f * dft.loc[i, current]
                i = i + 1
        if c0 > 0.5:
            smooth[ir] = c1 / c0
        else:
            #             ozcur[ir] = dft.loc[ir, 'Ifast_minib0_deconv']
            smooth[ir] = dft.loc[ir, current]

    dft['current_out'] = smooth

    return dft['current_out']

def smooth_parabolic_test(dft, current, time, tau):

    n2 = len(dft)
    n = tau + 1
    smooth = [0.] * n2

    for ir in range(0, len(dft)):
        timeir = dft.loc[ir, 'Tsim']
        ir1 = np.argmin(abs(time - (timeir - n)))
        ir2 = np.argmin(abs(time - (timeir + n)))
        c0 = 0.
        c1 = 0.
        # i = ir1
        while ir1 <= ir2:
            f = 1. - ((float(time[ir1]) - float(timeir)) / tau) ** 2
            if f <= 0:
                ir1 = ir1 + 1
            else:
                c0 = c0 + f
                c1 = c1 + f * dft.loc[ir1, current]
                ir1 = ir1 + 1
        if c0 > 0.5:
            smooth[ir] = c1 / c0
        else:
            smooth[ir] = dft.loc[ir, current]

    dft['current_out'] = smooth

    return dft['current_out']


def convolution(df, variable1, variable2, tvariable, beta, boolib0, ecctype):

    if ecctype == 'SPC': tfast = tfast_spc
    if ecctype == 'ENSCI': tfast = tfast_ecc

    size = len(df)
    Islow = [0]*size; Islow_conv = [0]*size; Ifast = [0]*size;
    Ifastminib0 = [0]*size; Ifast_deconv = [0]*size;
    Ifastminib0_deconv=[0]*size

    for i in range(size - 1):

        t1 = df.at[i + 1, tvariable]
        t2 = df.at[i, tvariable]

        Xs = np.exp(-(t1 - t2) / tslow)
        Xf = np.exp(-(t1 - t2) / tfast)

        Islow[i] = beta * df.at[i, variable1]
        Islow[i + 1] = beta * df.at[i + 1, variable1]
        Islow_conv[i + 1] =  Islow[i + 1] - (Islow[i + 1] - Islow_conv[i]) * Xs

        Ifast[i + 1] = af * (df.at[i + 1, variable2] - Islow_conv[i + 1])
        Ifast[i] = af * (df.at[i, variable2] - Islow_conv[i])
        Ifast_deconv[i + 1] = (Ifast[i + 1] - Ifast[i] * Xf) / (1 - Xf)

        if boolib0 == True:
            Ifastminib0[i] = af * (df.at[i, variable2] - Islow_conv[i] - df.at[i, 'iB0'])
            Ifastminib0[i + 1] = af * (df.at[i + 1, variable2] - Islow_conv[i + 1] - df.at[i + 1, 'iB0'])
            Ifastminib0_deconv[i + 1] = (Ifastminib0[i + 1] - Ifastminib0[i] * Xf) / (1 - Xf)


    if boolib0:
        return Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv
    else:
        return Islow, Islow_conv, Ifast, Ifast_deconv
    
def convolution_df(df, variable1, variable2, tvariable, beta, ecctype):

    if ecctype == 'SPC': tfast = tfast_spc
    if ecctype == 'ENSCI': tfast = tfast_ecc

    size = len(df)
    Islow = [0]*size; Islow_conv = [0]*size; Ifastminib0 = [0]*size;  
    Ifastminib0_deconv_out=[0]*size

    for i in range(size - 1):

        t1 = df.at[i + 1, tvariable]
        t2 = df.at[i, tvariable]

        Xs = np.exp(-(t1 - t2) / tslow)
        Xf = np.exp(-(t1 - t2) / tfast)

        Islow[i] = beta * df.at[i, variable1]
        Islow[i + 1] = beta * df.at[i + 1, variable1]
        Islow_conv[i + 1] =  Islow[i + 1] - (Islow[i + 1] - Islow_conv[i]) * Xs

        Ifastminib0[i] = af * (df.at[i, variable2] - Islow_conv[i] - df.at[i, 'iB0'])
        Ifastminib0[i + 1] = af * (df.at[i + 1, variable2] - Islow_conv[i + 1] - df.at[i + 1, 'iB0'])
        Ifastminib0_deconv_out[i + 1] = (Ifastminib0[i + 1] - Ifastminib0[i] * Xf) / (1 - Xf)

    df['Ifastminib0_deconv_out'] = Ifastminib0_deconv_out
    return df['Ifastminib0_deconv_out']




def convolution_specific_tau(df, variable1, variable2, tvariable, beta, boolib0, ecctype):

    if ecctype == 'SPC': tfast = tfast_spc
    if ecctype == 'ENSCI': tfast = tfast_ecc

    size = len(df)
    Islow = [0]*size; Islow_conv = [0]*size; Ifast = [0]*size; Ifastminib0 = [0]*size; Ifast_deconv = [0]*size; Ifastminib0_deconv=[0]*size
    Islow_conv_20 = [0] * size; Islow_conv_30 = [0] * size
    Ifastminib0_deconv_plus =  [0] * size; Ifastminib0_deconv_minus =  [0] * size;



    for i in range(size - 1):

        t1 = df.at[i + 1, tvariable]
        t2 = df.at[i, tvariable]

        Xs = np.exp(-(t1 - t2) / tslow)
        Xs_20 = np.exp(-(t1 - t2) / tslow_20)
        Xs_30 = np.exp(-(t1 - t2) / tslow_30)

        Xf = np.exp(-(t1 - t2) / tfast)
        Xf_p = np.exp(-(t1 - t2) / (tfast+4))
        Xf_m = np.exp(-(t1 - t2) / (tfast-4))


        Islow[i] = beta * df.at[i, variable1]
        Islow[i + 1] = beta * df.at[i + 1, variable1]
        Islow_conv[i + 1] =  Islow[i + 1] - (Islow[i + 1] - Islow_conv[i]) * Xs
        Islow_conv_20[i + 1] =  Islow[i + 1] - (Islow[i + 1] - Islow_conv[i]) * Xs_20
        Islow_conv_30[i + 1] =  Islow[i + 1] - (Islow[i + 1] - Islow_conv[i]) * Xs_30

        Ifast[i + 1] = af * (df.at[i + 1, variable2] - Islow_conv[i + 1])
        Ifast[i] = af * (df.at[i, variable2] - Islow_conv[i])
        Ifast_deconv[i + 1] = (Ifast[i + 1] - Ifast[i] * Xf) / (1 - Xf)

        if boolib0 == True:
            Ifastminib0[i] = af * (df.at[i, variable2] - Islow_conv[i] - df.at[i, 'iB0'])
            Ifastminib0[i + 1] = af * (df.at[i + 1, variable2] - Islow_conv[i + 1] - df.at[i + 1, 'iB0'])
            Ifastminib0_deconv[i + 1] = (Ifastminib0[i + 1] - Ifastminib0[i] * Xf) / (1 - Xf)
            Ifastminib0_deconv_plus[i + 1] = (Ifastminib0[i + 1] - Ifastminib0[i] * Xf_p) / (1 - Xf_p)
            Ifastminib0_deconv_minus[i + 1] = (Ifastminib0[i + 1] - Ifastminib0[i] * Xf_m) / (1 - Xf_m)



    if boolib0:
        return Islow_conv, Islow_conv_20, Islow_conv_30, Ifastminib0_deconv, \
               Ifastminib0_deconv_minus, Ifastminib0_deconv_plus
    else:
        return Islow, Islow_conv, Ifast, Ifast_deconv



def convolution_test(df, variable1, variable2, tvariable, beta, boolib0):

    size = len(df)
    Islow = [0]*size; Islow_conv = [0]*size; Ifast = [0]*size; Ifastminib0 = [0]*size; Ifast_deconv = [0]*size; Ifastminib0_deconv=[0]*size


    for i in range(size - 1):

        t1 = df.at[i + 1, tvariable]
        t2 = df.at[i, tvariable]

        Xs = np.exp(-(t1 - t2) / tslow)
        Xf = np.exp(-(t1 - t2) / tfast)

        Islow[i] = beta * df.at[i, variable1]
        Islow[i + 1] = beta * df.at[i + 1, variable1]

        Islow_conv[i + 1] = Islow[i + 1] - (Islow[i + 1] - Islow_conv[i]) * Xs

        Ifast[i + 1] = af * (df.at[i + 1, variable2])
        Ifast[i] = af * (df.at[i, variable2])

        Ifast_deconv[i + 1] = (Ifast[i + 1] - Ifast[i] * Xf) / (1 - Xf)

        if boolib0 == True:
            Ifastminib0[i] = af * (df.at[i, variable2] - df.at[i, 'iB0'])
            Ifastminib0[i + 1] = af * (df.at[i + 1, variable2]- df.at[i + 1, 'iB0'])
            Ifastminib0_deconv[i + 1] = (Ifastminib0[i + 1] - Ifastminib0[i] * Xf) / (1 - Xf)


    if boolib0:
        return Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv
    else:
        return Islow, Islow_conv, Ifast, Ifast_deconv


def smooth_and_convolute(df, variable, tvariable, windowlen,  beta):

    size = len(df)
    Islow = [0]*size; Islow_conv = [0]*size; Ifast = [0]*size; Ifastminib0 = [0]*size; Ifast_deconv = [0]*size; Ifastminib0_deconv=[0]*size


    df['variable_smoothed'] = df[variable].rolling(window=windowlen, center=True).mean()
    step = int(windowlen/2)
    for i in range(step, size - step):

        t1 = df.at[i + 1, tvariable]
        t2 = df.at[i, tvariable]

        Xs = np.exp(-(t1 - t2) / tslow)
        Xf = np.exp(-(t1 - t2) / tfast)

        Islow[i] = beta * df.at[i, variable]
        Islow[i + 1] = beta * df.at[i + 1, variable]
        Islow_conv[i + 1] = Islow[i + 1] - (Islow[i + 1] - Islow_conv[i]) * Xs
        Ifast[i + 1] = af * (df.at[i + 1, 'variable_smoothed'] - Islow_conv[i + 1])
        Ifast[i] = af * (df.at[i, 'variable_smoothed'] - Islow_conv[i])
        Ifast_deconv[i + 1] = (Ifast[i + 1] - Ifast[i] * Xf) / (1 - Xf)

        Ifastminib0[i] = af * (df.at[i, 'variable_smoothed'] - Islow_conv[i] - df.at[i, 'iB0'])
        Ifastminib0[i + 1] = af * (df.at[i + 1, 'variable_smoothed'] - Islow_conv[i + 1] - df.at[i + 1, 'iB0'])
        Ifastminib0_deconv[i + 1] = (Ifastminib0[i + 1] - Ifastminib0[i] * Xf) / (1 - Xf)

    return Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv



def convolution_pre(df, I0_var, variable1, variable2, tvariable, beta, boolib0):

    size = len(df)
    Islow = [0]*size; Islow_conv = [0]*size; Ifast = [0]*size; Ifastminib0 = [0]*size; Ifast_deconv = [0]*size; Ifastminib0_deconv=[0]*size


    for i in range(size - 1):

        t1 = df.at[i + 1, tvariable]
        t2 = df.at[i, tvariable]

        Xs = np.exp(-(t1 - t2) / tslow)
        Xf = np.exp(-(t1 - t2) / tfast)

        #
        # print('test1 ', df.at[i, variable1])
        # print('test2 ', I0_var)
        #       # * np.exp(-(t2)/tslow))
        # Islow[i] = beta * df.at[i, variable1] + I0_var * np.exp(-(t2) / tslow)
        Islow[i] = beta * df.at[i, variable1] + I0_var

        Islow[i + 1] = beta * df.at[i + 1, variable1] + I0_var

        # print(i, t1, t2, Islow[i], Islow[i+1])

        Islow_conv[i + 1] = (Islow[i + 1] - (Islow[i + 1] - Islow_conv[i]) * Xs)

        Ifast[i + 1] = af * (df.at[i + 1, variable2] - Islow_conv[i + 1])
        Ifast[i] = af * (df.at[i, variable2] - Islow_conv[i])

        Ifast_deconv[i + 1] = (Ifast[i + 1] - Ifast[i] * Xf) / (1 - Xf)

        if boolib0 == True:
            Ifastminib0[i] = af * (df.at[i, variable2] - Islow_conv[i] - df.at[i, 'iB0'])
            Ifastminib0[i + 1] = af * (df.at[i + 1, variable2] - Islow_conv[i + 1] - df.at[i + 1, 'iB0'])
            Ifastminib0_deconv[i + 1] = (Ifastminib0[i + 1] - Ifastminib0[i] * Xf) / (1 - Xf)


    if boolib0:
        return Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv
    else:
        return Islow, Islow_conv, Ifast, Ifast_deconv

def convolution_islow0(df, I0_var, variable1, variable2, tvariable, beta, boolib0, ecctype):

    if ecctype == 'SPC': tfast = tfast_spc
    if ecctype == 'ENSCI': tfast = tfast_ecc

    size = len(df)
    Islow = [0]*size; Islow_conv = [0]*size; Ifast = [0]*size; Ifastminib0 = [0]*size; Ifast_deconv = [0]*size; Ifastminib0_deconv=[0]*size


    for i in range(size - 1):

        t1 = df.at[i + 1, tvariable]
        t2 = df.at[i, tvariable]

        Xs = np.exp(-(t1 - t2) / tslow)
        Xf = np.exp(-(t1 - t2) / tfast)

        # version i sue dup to now, why seems wrong
        # Islow[i] = beta * df.at[i, variable1] + I0_var
        #test function updated version
        Islow[i] = beta * df.at[i, variable1]

        Islow[i + 1] = beta * df.at[i + 1, variable1]
        # print(i, t1, t2, Islow[i], Islow[i+1])
        Islow_conv[0] = I0_var
        # print('in function', I0_var)

        Islow_conv[i + 1] = Islow[i + 1] - (Islow[i + 1] - Islow_conv[i]) * Xs

        Ifast[i + 1] = af * (df.at[i + 1, variable2] - Islow_conv[i + 1])
        Ifast[i] = af * (df.at[i, variable2] - Islow_conv[i])

        Ifast_deconv[i + 1] = (Ifast[i + 1] - Ifast[i] * Xf) / (1 - Xf)

        if boolib0 == True:
            Ifastminib0[i] = af * (df.at[i, variable2] - Islow_conv[i] - df.at[i, 'iB0'])
            Ifastminib0[i + 1] = af * (df.at[i + 1, variable2] - Islow_conv[i + 1] - df.at[i + 1, 'iB0'])
            Ifastminib0_deconv[i + 1] = (Ifastminib0[i + 1] - Ifastminib0[i] * Xf) / (1 - Xf)


    if boolib0:
        return Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv
    else:
        return Islow, Islow_conv, Ifast, Ifast_deconv

def smooth_convolution_islow0(df, I0_var, variable1, variable2, tvariable, beta, windowlen):

    size = len(df)
    Islow = [0]*size; Islow_conv = [0]*size; Ifast = [0]*size; Ifastminib0 = [0]*size; Ifast_deconv = [0]*size; Ifastminib0_deconv=[0]*size

    df[variable2] = df[variable2].rolling(window=windowlen, center=True).mean()

    step = int(windowlen / 2)

    for i in range(step, size - step):

        t1 = df.at[i + 1, tvariable]
        t2 = df.at[i, tvariable]

        Xs = np.exp(-(t1 - t2) / tslow)
        Xf = np.exp(-(t1 - t2) / tfast)

        # version i sue dup to now, why seems wrong
        # Islow[i] = beta * df.at[i, variable1] + I0_var
        Islow_conv[0] = I0_var
        #test function updated version
        Islow[i] = beta * df.at[i, variable1]
        Islow[i + 1] = beta * df.at[i + 1, variable1]
        # print(i, t1, t2, Islow[i], Islow[i+1])

        Islow_conv[i + 1] = Islow[i + 1] - (Islow[i + 1] - Islow_conv[i]) * Xs
        Ifast[i + 1] = af * (df.at[i + 1, variable2] - Islow_conv[i + 1])
        Ifast[i] = af * (df.at[i, variable2] - Islow_conv[i])
        Ifast_deconv[i + 1] = (Ifast[i + 1] - Ifast[i] * Xf) / (1 - Xf)

        Ifastminib0[i] = af * (df.at[i, variable2] - Islow_conv[i] - df.at[i, 'iB0'])
        Ifastminib0[i + 1] = af * (df.at[i + 1, variable2] - Islow_conv[i + 1] - df.at[i + 1, 'iB0'])
        Ifastminib0_deconv[i + 1] = (Ifastminib0[i + 1] - Ifastminib0[i] * Xf) / (1 - Xf)

    return Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv



def convolution_islow0_test(df, I0_var, variable1, variable2, tvariable, beta, boolib0):

    size = len(df)
    Islow = [0]*size; Islow_conv = [0]*size; Ifast = [0]*size; Ifastminib0 = [0]*size; Ifast_deconv = [0]*size; Ifastminib0_deconv=[0]*size

    # Islow_conv[0] = I0_var

    for i in range(size - 1):

        t1 = df.at[i + 1, tvariable]
        t2 = df.at[i, tvariable]

        Xs = np.exp(-(t1 - t2) / tslow)
        Xf = np.exp(-(t1 - t2) / tfast)


        Islow[i] = beta * df.at[i, variable1]

        Islow[i + 1] = beta * df.at[i + 1, variable1]
        # print(i, t1, t2, Islow[i], Islow[i+1])
        Islow_conv[0] = I0_var

        Islow_conv[i + 1] = Islow[i + 1] - (Islow[i + 1] - Islow_conv[i]) * Xs

        Ifast[i + 1] = af * (df.at[i + 1, variable2] - Islow_conv[i + 1])
        Ifast[i] = af * (df.at[i, variable2] - Islow_conv[i])

        Ifast_deconv[i + 1] = (Ifast[i + 1] - Ifast[i] * Xf) / (1 - Xf)

        if boolib0 == True:
            Ifastminib0[i] = af * (df.at[i, variable2] - Islow_conv[i] - df.at[i, 'iB0'])
            Ifastminib0[i + 1] = af * (df.at[i + 1, variable2] - Islow_conv[i + 1] - df.at[i + 1, 'iB0'])
            Ifastminib0_deconv[i + 1] = (Ifastminib0[i + 1] - Ifastminib0[i] * Xf) / (1 - Xf)


    if boolib0:
        return Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv
    else:
        return Islow, Islow_conv, Ifast, Ifast_deconv
