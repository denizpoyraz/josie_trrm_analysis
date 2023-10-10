import numpy as np
import pandas as pd
from scipy import stats
# Libraries
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from data_cuts import cuts0910, cuts2017, cuts9602
from homogenization_functions import pumpflow_efficiency, return_phipcor, VecInterpolate_log
from constant_variables import *
from analyse_functions import Calc_average_profile_pressure, calc_average_df_pressure, set_columns_nopair_dependence
from plotting_functions import filter_rdif, filter_rdif_all

df17c = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_deconv_2023paper.csv", low_memory=False)
df17c = df17c[df17c.iB2 > -9]
df09c = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/"
                    "Josie0910_deconv_2023_unitedpaper.csv", low_memory=False)


df09 = cuts0910(df09c)
df17 = cuts2017(df17c)
# df96 = cuts9602(df96c)


df09['Year'] = '0910'
df17['Year'] = '2017'
# df96['Year'] = '9602'

# dfa = pd.concat([df09, df17, df96], ignore_index=True)
dfa = pd.concat([df09, df17], ignore_index=True)

df = df09
df = df[(df.Sim == 136) & (df.Team ==1)]

df = df.reset_index()
for k in range(len(df)):

    for p in range(len(JMA) - 1):

        if (df.at[k, 'Pair'] >= Pval_jma[p + 1]) & (df.at[k, 'Pair'] < Pval_jma[p]):
            df.at[k, 'I_OPM_jma_pr'] = df.at[k, 'PO3_OPM'] * df.at[k, 'PFcor']  / \
                                    (df.at[k, 'Tpump_cor'] * 0.043085 * JMA[p])

    if (df.at[k, 'Pair'] <= Pval_jma[-1]):
        df.at[k, 'I_OPM_jma_pr'] = df.at[k, 'PO3_OPM'] * df.at[k, 'PFcor'] / \
                                (df.at[k, 'Tpump_cor'] * 0.043085  * JMA[-1])



column_list = ['Pair','Tsim','IM','Tpump_cor','PO3','PO3_dqa','I_OPM_jma_pr','I_OPM_jma']


yrefd = [1000, 850, 700, 550, 400, 350, 300, 200, 150, 100, 75, 50, 35,30, 25, 15,9, 8, 6]

dft = calc_average_df_pressure(df, column_list, yrefd)

fig = plt.figure(figsize=(15, 12))
plt.plot(dft.I_OPM_jma, dft.Pair, label = 'new jma interpolation')
plt.plot(dft.I_OPM_jma_pr, dft.Pair, label = 'previous method no inter.')
plt.legend(loc='upper right', frameon=True, fontsize='x-large', markerscale=1,  handletextpad=0.1)

plt.show()

