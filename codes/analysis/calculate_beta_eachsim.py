import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from beta_functions import ratiofunction_beta_new, ratiofunction_beta_9602_new
from data_cuts import cuts0910_beta, cuts2017, cuts9602
from analyse_functions import filter_df, filter20
from constant_variables import *


######################################################################################################################


df1 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Data_2023paper.csv", low_memory=False)
df1 = df1[df1.ADX == 0]

#
# df1 = df1.drop(df1[(df1.Sim == 147) & (df1.Team == 3)].index)
# df1 = df1.drop(df1[(df1.Sim == 158) & (df1.Team == 1)].index)
# df1 = df1.drop(df1[(df1.Sim == 158) & (df1.Team == 2)].index)
# df1 = df1.drop(df1[(df1.Sim == 160) & (df1.Team == 4)].index)
# df1 = df1.drop(df1[(df1.Sim == 165) & (df1.Team == 4)].index)

df2 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie9602_Data_bkg.csv", low_memory=False)

df2 = cuts9602(df2)


sim_0910, team_0910 = filter_df(df1)
sim_9602, team_9602 = filter_df(df2)

sim_9602_20, team_9602_20 = filter20(df2)
#

# ## for 0910 

rmean_en0505_0910, rstd_en0505, rmedian_en0505_0910, rqerr_en0505_0910, df1 = ratiofunction_beta_new\
    (df1, sim_0910[0], team_0910[0], 'EN0505', 1, tslow)
rmean_en1010_0910, rstd_en1010, rmedian_en1010_0910, rqerr_en1010_0910,df1 = ratiofunction_beta_new\
    (df1, sim_0910[1], team_0910[1], 'EN1010', 1, tslow)
rmean_sp0505_0910, rstd_sp0505, rmedian_sp0505_0910, rqerr_sp0505_0910, df1 = ratiofunction_beta_new\
    (df1, sim_0910[2], team_0910[2], 'SP0505', 1, tslow)
rmean_sp1010_0910, rstd_sp1010, rmedian_sp1010_0910, rqerr_sp1010_0910, df1 = ratiofunction_beta_new\
    (df1, sim_0910[3], team_0910[3], 'SP1010', 1, tslow)

# df1.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Data_nocut_beta_tslow30.csv")
df1.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Data_nocut_beta_paper.csv")

# print("0910 EN0505")
# print(np.nanmedian(rmedian_en0505_0910))
# # print(rqerr_en0505)


rmean_en0505_9602, rstd_en0505_9602, rmedian_en0505_9602,  rqerr_en0505_9602, df2 = ratiofunction_beta_9602_new\
    (df2, sim_9602[0], team_9602[0], 'EN0505', 1, tslow)
rmean_en1010_9602, rstd_en1010_9602, rmedian_en1010_9602, rqerr_en1010_9602, df2 = ratiofunction_beta_9602_new\
    (df2, sim_9602[1], team_9602[1], 'EN1010', 1, tslow)
rmean_sp0505_9602, rstd_sp0505_9602, rmedian_sp0505_9602, rqerr_sp0505_9602, df2 = ratiofunction_beta_9602_new\
    (df2, sim_9602[2], team_9602[2], 'SP0505', 1, tslow)
rmean_sp1010_9602, rstd_sp1010_9602, rmedian_sp1010_9602, rqerr_sp1010_9602, df2 = ratiofunction_beta_9602_new\
    (df2, sim_9602[3], team_9602[3], 'SP1010', 1, tslow)
#

df2.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie9602_Data_withcut_beta.csv")






