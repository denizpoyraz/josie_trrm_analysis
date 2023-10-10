#test smoothing
import numpy as np
import pandas as pd
from scipy import stats
# Libraries
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from data_cuts import cuts0910, cuts2017, cuts9602
from plotting_functions import filter_rdif, filter_rdif_all
from analyse_functions import Calc_average_Dif_yref,apply_calibration, cal_dif
from constant_variables import *
from convolution_functions import smooth_gaussian,  convolution_df
import warnings
warnings.filterwarnings("ignore")



# df = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie9602_calibrated.csv", low_memory=False)
# dft = df[(df.Sim == 100) & (df.Team ==6)]

df = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_deconv_2023_unitedpaper.csv")
# dft = df[(df.Sim == 136) & (df.Team ==1)]
# beta = beta_sp1010
# tfast = tfast_spc
dft = df[(df.Sim == 136) & (df.Team ==3)]
beta = beta_en1010
tfast = tfast_ecc
dft = dft.reset_index()
dft['IminusiB0'] = dft['IM'] - dft['iB0']

sigma = 0.15*tfast
timew = 3*sigma

sigma2 = 0.2*tfast
timew2 = 3*sigma2

dft['I_gsm'] = smooth_gaussian(dft,'Tsim', 'IM', timew, sigma, 'I_gsm')
dft['IminiB0_gsm'] = smooth_gaussian(dft,'Tsim', 'IminusiB0', timew, sigma, 'IminiB0_gsm')



dft['I_rm10'] = dft['IM'].rolling(window=5, center=True).mean()
dft['IminusiB0_rm10'] = dft['IminusiB0'].rolling(window=5, center=True).mean()



dft = dft[5:]
dft = dft.reset_index()


dft['Ifast_minib0_deconv_gs'] = convolution_df(dft, 'IminiB0_gsm', 'I_gsm', 'Tsim', beta, 'ENSCI')
# dft['Ifast_minib0_deconv_gs2'] = convolution_df(dft, 'IminiB0_gsm2', 'I_gsm2', 'Tsim', beta, 'ENSCI')

dft['Ifast_minib0_deconv_rm10'] = convolution_df(dft, 'IminusiB0_rm10', 'I_rm10', 'Tsim', beta, 'ENSCI')

# dft['I2fast_minib0_deconv_gs'] = convolution_df(dft, 'I2miniB0_gsm', 'I2_gsm', 'Tsim', beta, 'ENSCI')


size_label = 22
size_title = 24
size_text = 18

plt.figure(figsize=(12, 8))
plt.ylabel(r'Current [$\mu$A])', fontsize=size_label)
plt.xlabel(r'Simulation time $[sec.]$', fontsize = size_label)
plt.yticks(fontsize=size_label)
plt.xticks(fontsize=size_label)
plt.gca().yaxis.set_major_formatter(ScalarFormatter())

plt.gca().tick_params(which='major', width=2)
plt.gca().tick_params(which='minor', width=2)
# plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().xaxis.set_tick_params(length=5, which='minor')
plt.gca().xaxis.set_tick_params(length=10, which='major')
# plt.gca().yaxis.set_tick_params(length=5, which='minor')
plt.gca().yaxis.set_tick_params(length=10, which='major')

plt.ylim(0.001, 10)
plt.xlim(0, 8500)
# plt.yscale('log')

ts = df.Tsim.shift()[2]
print('ts', ts)
ws = int(10/ts)
print('ws,', ws)



dft['Ifast_minib0_deconv_sm20'] = dft['Ifast_minib0_deconv'].rolling(window=2*ws, center=True).mean()
dft['Ifast_minib0_deconv_sm10'] = dft['Ifast_minib0_deconv'].rolling(window=ws, center=True).mean()



#
# #fig 6
pre = 'fig_6_'
# plt.plot(dft.Tsim, dft.IM - dft.at[0,'iB0'], label = r'I$_M$ - I$_{B0}$', color =cbl[4]) #blue
plt.plot(dft.Tsim, dft.I_OPM_jma, label=r'I$_{OPM}$')  # red
# plt.plot(dft.Tsim, dft.Ifast_minib0_deconv_ib1_decay, label=r'I$_{F,D}$', color = cbl[2]) #yellow
#
# # plt.plot(dft.Tsim, dft.i_fast_par_sm10, label=r'I$_{F,D}$ (parabolic smoothing 10 sec.)', color = cbl[3]) #red
# plt.plot(dft.Tsim, dft.Ifast_minib0_deconv_sm5, label=r'I$_{F,D}$ (running averages 5 sec.)', color = cbl[1]) #red
plt.plot(dft.Tsim, dft.Ifast_minib0_deconv_sm10, label=r'I$_{F,D}$ (running averages 10 sec., after)')  # red
plt.plot(dft.Tsim, dft.Ifast_minib0_deconv_gs, label=fr'I$_F$ (gaussian method {timew}, {sigma} - smoothed before)')  # red
# plt.plot(dft.Tsim, dft.Ifast_minib0_deconv_gs2, label=fr'I$_F$ (gaussian method {timew2}, {sigma2} - smoothed before)')  # red
# plt.plot(dft.Tsim, dft.Ifast_minib0_deconv_rm10, label=r'I$_{F,D}$ (running averages 10 sec., before)')  # red
# plt.plot(dft.Tsim, dft.I2fast_minib0_deconv_gs, label=fr'I$_F$ (test gaussian method {timew}, {sigma} - smoothed before)')  # red

# plt.ylim(-0.15, 0.9)
# plt.xlim(1900, 2600)
# plt.ylim(-0.05, 0.9)
# plt.xlim(1700, 2600)
# cbl = ['#e41a1c', '#a65628', '#dede00', '#4daf4a', '#377eb8', '#984ea3']
# red, brown, yellow, green, blue, purple

# df['i_fast_par_sm10'] = smooth_parabolic(df, 'Ifast_minib0_deconv', df.Tsim, 5)
# df['i_fast_par_sm20'] = smooth_parabolic(df, 'Ifast_minib0_deconv', df.Tsim, 10)

# df['Ifast_minib0_deconv_sm5'] = df['Ifast_minib0_deconv'].rolling(window=3, center=True).mean()

plt.legend(loc='upper left', frameon=True, fontsize='large', markerscale=10, handletextpad=0.1)

plt.show()