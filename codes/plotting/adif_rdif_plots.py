# Libraries
import pandas as pd
import matplotlib.pyplot as plt

from  analyse_functions import  Calc_average_Dif_yref
from  plotting_functions import errorPlot_ARDif_withtext, errorPlot_ARDif
from  data_cuts import cuts0910, cuts2017, cuts9602
from  plotting_functions import filter_rdif
from constant_variables import *

folderpath = ''

year_2017 = False
year_9602 = True
year_0910 = False

bool_current = 'True'
bool_pressure = 'True'
if year_9602:
    df = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie9602_calibrated.csv", low_memory=False)

if year_2017:
    df = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2017_calibrated.csv", low_memory=False)

if year_0910:

    df = pd.read_csv(
            "/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_calibrated.csv",low_memory=False)
    print(list(df))



if year_9602 or year_2017:
    df['Ifast_minib0_deconv_sm10'] = df['Ifast_minib0_deconv'].rolling(window=5, center=True).mean()

if year_0910:

    df['Ifast_minib0_deconv_sm10'] = df['Ifast_minib0_deconv_ib1_decay'].rolling(window=5, center=True).mean()


if year_0910:
    df = cuts0910(df)
if year_2017:
    df = cuts2017(df)
if year_9602:
    df = cuts9602(df)

df['IminusiB1'] = df['IM'] - df['iB1']
if year_2017:
    df['IminusiB1'] = df['IM'] - df['iB2']
    filt_01 = (df.Buf == 0.1)
    not_filt_01 = (df.Buf != 0.1)
    df.loc[filt_01, 'I_OPM_kom'] = (df.loc[filt_01, 'PO3_OPM'] * df.loc[filt_01, 'PFcor_jma']) / \
                                   (df.loc[filt_01, 'Tpump_cor'] * 0.043085)
    df.loc[not_filt_01, 'I_OPM_kom'] = (df.loc[not_filt_01, 'PO3_OPM'] * df.loc[not_filt_01, 'PFcor_kom']) / \
                                   (df.loc[not_filt_01, 'Tpump_cor'] * 0.043085)

    # df.loc[filt_01, 'PO3_dqa'] = (0.043085 * df.loc[filt_01, 'Tpump_cor'] * (
    #             df.loc[filt_01, 'IM'] - df.loc[filt_01, 'iB2'])) \
    #                              / (df.loc[filt_01, 'PFcor_jma'])
    # df.loc[not_filt_01, 'PO3_dqa'] = (0.043085 * df.loc[not_filt_01, 'Tpump_cor'] * (
    #             df.loc[not_filt_01, 'IM'] - df.loc[not_filt_01, 'iB2'])) \
    #                                  / (df.loc[not_filt_01, 'PFcor_kom'])

df['PO3_cal'] = (0.043085 * df['Tpump_cor'] * df['I_corrected']) / (df['PFcor_jma'])
if year_9602:
    df['PO3_cor'] = (0.043085 * df['Tpump_cor'] * df['Ifast_minib0_deconv_sm10']) / (df['PFcor_jma'])
    
###############################################
# Filters for Sonde, Solution, Buff er selection
prof = filter_rdif(df, year_9602, year_0910, year_2017)

print(list(df))
# ##################################################################################
# ################     PLOTS        #################################
# ##################################################################################

ytitle = 'Pressure [hPa]'
# 2009
labellist = ['EN-SCI SST0.5', 'EN-SCI SST1.0', 'SPC SST0.5', 'SPC SST1.0']
# 2017
if year_2017:
    labellist = ['EN-SCI SST0.5', 'EN-SCI SST0.1', 'SPC SST0.1', 'SPC SST1.0']

# ### Plotting

# First do the plotting using PO3
rxtitle = 'Relative Differences [%] \n (Sonde - OPM)/OPM'
# axtitle = 'Sonde - OPM [mPa]'
axtitlecur = r'Sonde - OPM [$\mu$A]'

title = 'JOSIE 0910'
if year_2017:
    title = 'JOSIE 2017'
    y = '2017'
if year_9602:
    y = '9602'
    title = 'JOSIE 1996/1998/2000/2002'
if year_0910:
    y = '0910'
    title = 'JOSIE 2009/2010'
    
ptitle = f'Conventional {title}' 
ptitle_trrm = f'TRRM {title}' 
ptitle_cal = f'Calibrated {title}' 

# ##################################################################################
# # ################      CURRENT IM PLOTS        #################################
# # ##################################################################################
# standard for all years
adif_IM, adif_IM_err, rdif_IM, rdif_IM_err, Yp =  Calc_average_Dif_yref(prof, 'IminusiB1', 'I_OPM_kom', 'pressure',yref)
adif_IM_trrm, adif_IM_trrm_err, rdif_IM_trrm, rdif_IM_trrm_err, Yp = \
     Calc_average_Dif_yref(prof, 'Ifast_minib0_deconv_sm10', 'I_OPM_jma', 'pressure', yref)
adif_IM_cal, adif_IM_cal_err, rdif_IM_cal, rdif_IM_cal_err, Yp = \
     Calc_average_Dif_yref(prof, 'I_corrected', 'I_OPM_jma', 'pressure', yref)

nsim = [0] * 4
for i in range(len(prof)):
    nsim[i] = len(prof[i].drop_duplicates(['Sim', 'Team']))
    print('nsim', nsim[i])

adifc = f'ADif_{y}_current_conventional'
adift_c = f'ADif_{y}_current_trrm'
adifc_c = f'ADif_{y}_current_calibrated'
rdifc = f'RDif_{y}_current_conventional'
rdift_c = f'RDif_{y}_current_trrm'
rdifc_c = f'RDif_{y}_current_calibrated'

if bool_current:


    errorPlot_ARDif_withtext(adif_IM, adif_IM_err, Yp, [-0.7, 0.7], [1000, 5], ptitle,
                             axtitlecur, ytitle, labellist, adifc, nsim, True,1)

    errorPlot_ARDif_withtext(adif_IM_trrm, adif_IM_trrm_err, Yp, [-0.7, 0.7], [1000, 5], ptitle_trrm,
                                 axtitlecur, ytitle, labellist, adift_c,nsim,True, 1)

    errorPlot_ARDif_withtext(adif_IM_cal, adif_IM_cal_err, Yp, [-0.7, 0.7], [1000, 5], ptitle_cal,
                                 axtitlecur, ytitle, labellist, adifc_c,nsim,True, 1)

    errorPlot_ARDif_withtext(rdif_IM, rdif_IM_err, Yp, [-40, 40], [1000, 5], ptitle,
                                 rxtitle, ytitle, labellist, rdifc, nsim, True, 1)

    errorPlot_ARDif_withtext(rdif_IM_trrm, rdif_IM_trrm_err, Yp, [-40, 40], [1000, 5], ptitle_trrm,
                                 rxtitle, ytitle, labellist, rdift_c, nsim,True, 1)

    errorPlot_ARDif_withtext(rdif_IM_cal, rdif_IM_cal_err, Yp, [-40, 40], [1000, 5], ptitle_cal,
                                 rxtitle, ytitle, labellist, rdifc_c, nsim,True, 1)



# ##################################################################################
# # ################      Pressure PLOTS        #################################
# # ##################################################################################
# standard for all years
adif_P, adif_P_err, rdif_P, rdif_P_err, Yp =  Calc_average_Dif_yref(prof, 'PO3_dqa', 'PO3_OPM', 'pressure',yref)
adif_P_trrm, adif_P_trrm_err, rdif_P_trrm, rdif_P_trrm_err, Yp = \
     Calc_average_Dif_yref(prof, 'PO3_cor', 'PO3_OPM', 'pressure', yref)
adif_P_cal, adif_P_cal_err, rdif_P_cal, rdif_P_cal_err, Yp = \
     Calc_average_Dif_yref(prof, 'PO3_cal', 'PO3_OPM', 'pressure', yref)

nsim = [0] * 4
for i in range(len(prof)):
    nsim[i] = len(prof[i].drop_duplicates(['Sim', 'Team']))
    print('nsim', nsim[i])

adifc = f'ADif_{y}_pressure_conventional'
adift_c = f'ADif_{y}_pressure_trrm'
adifc_c = f'ADif_{y}_pressure_calibrated'
rdifc = f'RDif_{y}_pressure_conventional'
rdift_c = f'RDif_{y}_pressure_trrm'
rdifc_c = f'RDif_{y}_pressure_calibrated'

axtitlepre = r'Sonde - OPM [mPa]'

if bool_pressure:


    errorPlot_ARDif_withtext(adif_P, adif_P_err, Yp, [-2,2], [1000, 5], ptitle,
                             axtitlepre, ytitle, labellist, adifc, nsim, True,1)

    errorPlot_ARDif_withtext(adif_P_trrm, adif_P_trrm_err, Yp, [-2,2], [1000, 5], ptitle_trrm,
                                 axtitlepre, ytitle, labellist, adift_c,nsim,True, 1)

    errorPlot_ARDif_withtext(adif_P_cal, adif_P_cal_err, Yp, [-2,2], [1000, 5], ptitle_cal,
                                 axtitlepre, ytitle, labellist, adifc_c,nsim,True, 1)

    errorPlot_ARDif_withtext(rdif_P, rdif_P_err, Yp, [-40, 40], [1000, 5], ptitle,
                                 rxtitle, ytitle, labellist, rdifc, nsim, True, 1)

    errorPlot_ARDif_withtext(rdif_P_trrm, rdif_P_trrm_err, Yp, [-40, 40], [1000, 5], ptitle_trrm,
                                 rxtitle, ytitle, labellist, rdift_c, nsim,True, 1)

    errorPlot_ARDif_withtext(rdif_P_cal, rdif_P_cal_err, Yp, [-40, 40], [1000, 5], ptitle_cal,
                                 rxtitle, ytitle, labellist, rdifc_c, nsim,True, 1)



