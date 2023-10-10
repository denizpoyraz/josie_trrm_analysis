import pandas as pd
from convolution_functions import convolution, smooth_gaussian
from constant_variables import *
import numpy as np

betal = [0.01 * n for n in range(1, 11)]
print(betal[0], betal[1], betal[8], betal[9])

#update of the code 25/04 for HV smoothing
beta_spe = False
bool_9602 = False
bool_0910_decay = False
bool_2017 = False
bool_sm_hv = True
bool_2beta = False

year = '0910'
if bool_sm_hv: pre = '_sm_hv_betascan'
if bool_2beta: pre = '_2beta'
if bool_2beta and bool_sm_hv: pre = '_sm_hv_2beta'

file_out = f'Josie{year}_deconv_2023paper_betascan.csv'
# file_out = f'Josie{year}_deconv_2023paper{pre}.csv'


df = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_Data_2023paper.csv", low_memory=False)
if bool_2017:
    df = pd.read_csv(f"/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie{year}_Data_2023paper_ib2.csv", low_memory=False)
    # file_out = f'Josie{year}_deconv_2023paper_ib2{pre}.csv'
    file_out = f'Josie{year}_deconv_2023paper_betascan.csv'


####            #######             ########


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
    # print(title, beta)

    if bool_9602: dft[j]['TPext'] = dft[j]['TPint']
    #update everthing to HV smoothed IM

    dft[j]['IM_gsm'] = smooth_gaussian(dft[j], 'Tsim', 'IM', timew, sigma)
    dft[j]['IMminusIB0_gsm'] = smooth_gaussian(dft[j], 'Tsim', 'IMminusIB0', timew, sigma)

    Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv = \
        convolution(dft[j], 'IMminusIB0', 'IM', 'Tsim', beta, 1, sondestr)
    if bool_sm_hv:
        Islow, Islow_conv, Ifast, Ifast_deconv, Ifastminib0, Ifastminib0_deconv = \
            convolution(dft[j], 'IMminusIB0_gsm', 'IM_gsm', 'Tsim', betal[0], 1, sondestr)
        results = []
        for beta in betal:
            result = convolution(dft[j], 'IMminusIB0_gsm', 'IM_gsm', 'Tsim', beta, 1, sondestr)
            results.append(result)

        # Islowa, Islow_conva, Ifasta, Ifast_deconva, Ifastminib0a, Ifastminib0_deconva = results

    for ij in range(0,10):
        islow=f'Islow_{ij}'
        islow_conv=f'Islow_conv_{ij}'
        ifast=f'Ifast_{ij}'
        ifast_minib0=f'Ifast_minib0_{ij}'
        ifast_deconv=f'Ifast_deconv_{ij}'
        ifast_minib0_deconv=f'Ifast_minib0_deconv_{ij}'
        ifast_minib0_deconv_sm10=f'Ifast_minib0_deconv_sm10_n{ij}'
        dft[j][islow] = results[ij][0]
        dft[j][islow_conv] = results[ij][1]
        dft[j][ifast] = results[ij][2]
        dft[j][ifast_deconv] = results[ij][3]
        dft[j][ifast_minib0] = results[ij][4]
        dft[j][ifast_minib0_deconv] = results[ij][5]
        dft[j][ifast_minib0_deconv_sm10] = dft[j][ifast_minib0_deconv]

    dft[j]['Islow'] = Islow
    dft[j]['Islow_conv'] = Islow_conv
    dft[j]['Ifast'] = Ifast
    dft[j]['Ifast_minib0'] = Ifastminib0
    dft[j]['Ifast_deconv'] = Ifast_deconv
    dft[j]['Ifast_minib0_deconv'] = Ifastminib0_deconv


    list_data.append(dft[j])

df_dc = pd.concat(list_data, ignore_index=True)

df_dc.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/" + file_out)
