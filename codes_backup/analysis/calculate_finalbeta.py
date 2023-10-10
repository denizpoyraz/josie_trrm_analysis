import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from data_cuts import cuts0910_beta, cuts9602
from analyse_functions import filter_solsonde



def print_round(x,y):
    print(round(x,3)," +/- ",round(y,3) )


one_beta_persim = True


df1 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Data_nocut_beta_paper.csv")
# df1 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Data_nocut_beta_paper_noib0.csv")

df2 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie9602_Data_withcut_beta.csv", low_memory=False)
df1 = df1[df1.ADX == 0]
df1 = df1.drop(df1[(df1.Sim == 147) & (df1.Team == 3)].index)
df1 = df1.drop(df1[(df1.Sim == 167) & (df1.Team == 4)].index)
df1 = df1.drop(df1[(df1.Sim == 158) & (df1.Team == 2)].index)
df1 = df1.drop(df1[(df1.Sim == 158) & (df1.Team == 1)].index)


df1.loc[df1.Sim==144,'beta4'] = -9
df1.loc[(df1.Sim==166) & (df1.Team==1),'beta2'] = -9

df1 = df1[df1.beta4 > 0]
df1 = df1[df1.beta2 > 0]

# df2 = cuts9602(df2)
df1 = df1[df1.ADX == 0]
df1 = df1.drop(df1[(df1.Sim == 147) & (df1.Team == 3)].index)
df1 = df1.drop(df1[(df1.Sim == 167) & (df1.Team == 4)].index)
df1 = df1.drop(df1[(df1.Sim == 158) & (df1.Team == 1)].index)
df1 = df1.drop(df1[(df1.Sim == 158) & (df1.Team == 2)].index)

df1 = df1[df1.iB0 >= 0]

df_0910_en0505, df_0910_en1010, df_0910_sp0505, df_0910_sp1010 = filter_solsonde(df1)
df_9602_en0505, df_9602_en1010, df_9602_sp0505, df_9602_sp1010 = filter_solsonde(df2)

if one_beta_persim:
    beta_0910_en0505 = np.array(df_0910_en0505.beta)
    beta_0910_en1010 = np.array(df_0910_en1010.beta)
    beta_0910_sp0505 = np.array(df_0910_sp0505.beta)
    beta_0910_sp1010 = np.array(df_0910_sp1010.beta)
    
    beta_9602_en0505 = np.array(df_9602_en0505.beta)
    beta_9602_en1010 = np.array(df_9602_en1010.beta)
    beta_9602_sp0505 = np.array(df_9602_sp0505.beta)
    beta_9602_sp1010 = np.array(df_9602_sp1010.beta)
    
if not one_beta_persim:
    beta1_0910_en0505 = np.array(df_0910_en0505.beta1)
    beta1_0910_en1010 = np.array(df_0910_en1010.beta1)
    beta1_0910_sp0505 = np.array(df_0910_sp0505.beta1)
    beta1_0910_sp1010 = np.array(df_0910_sp1010.beta1)

    beta2_0910_en0505 = np.array(df_0910_en0505.beta2)
    beta2_0910_en1010 = np.array(df_0910_en1010.beta2)
    beta2_0910_sp0505 = np.array(df_0910_sp0505.beta2)
    beta2_0910_sp1010 = np.array(df_0910_sp1010.beta2)

    beta3_0910_en0505 = np.array(df_0910_en0505.beta3)
    beta3_0910_en1010 = np.array(df_0910_en1010.beta3)
    beta3_0910_sp0505 = np.array(df_0910_sp0505.beta3)
    beta3_0910_sp1010 = np.array(df_0910_sp1010.beta3)

    beta4_0910_en0505 = np.array(df_0910_en0505.beta4)
    beta4_0910_en1010 = np.array(df_0910_en1010.beta4)
    beta4_0910_sp0505 = np.array(df_0910_sp0505.beta4)
    beta4_0910_sp1010 = np.array(df_0910_sp1010.beta4)

   

    beta_0910_en0505 = np.concatenate([beta1_0910_en0505, beta2_0910_en0505, beta3_0910_en0505, beta4_0910_en0505], axis=None)
    beta_0910_en1010 = np.concatenate([beta1_0910_en1010, beta2_0910_en1010, beta3_0910_en1010, beta4_0910_en1010], axis=None)
    beta_0910_sp0505 = np.concatenate([beta1_0910_sp0505, beta2_0910_sp0505, beta3_0910_sp0505, beta4_0910_sp0505], axis=None)
    beta_0910_sp1010 = np.concatenate([beta1_0910_sp1010, beta2_0910_sp1010, beta3_0910_sp1010, beta4_0910_sp1010], axis=None)

    beta_9602_en0505 = np.array(df_9602_en0505[['beta1', 'beta2']].values.tolist())
    beta_9602_en1010 = np.array(df_9602_en1010[['beta1', 'beta2']].values.tolist())
    beta_9602_sp0505 = np.array(df_9602_sp0505[['beta1', 'beta2']].values.tolist())
    beta_9602_sp1010 = np.array(df_9602_sp1010[['beta1', 'beta2']].values.tolist())
    

beta_en0505 = np.concatenate([beta_0910_en0505, beta_9602_en0505 ], axis= None)
beta_en1010 = np.concatenate([beta_0910_en1010, beta_9602_en1010 ], axis= None)
beta_sp0505 = np.concatenate([beta_0910_sp0505, beta_9602_sp0505 ], axis= None)
beta_sp1010 = np.concatenate([beta_0910_sp1010, beta_9602_sp1010 ], axis= None)

if one_beta_persim:
    print('One beta for each Sim')
if not one_beta_persim:
    print('All betas ')

e05_err = stats.median_abs_deviation(beta_0910_en0505)
e10_err = stats.median_abs_deviation(beta_0910_en1010)
s05_err = stats.median_abs_deviation(beta_0910_sp0505)
s10_err = stats.median_abs_deviation(beta_0910_sp1010)

print("0910: EN0505    EN1010    SP0505   SP1010")
print('size', len(beta_0910_en0505), len(beta_0910_en1010), len(beta_0910_sp0505), len(beta_0910_sp1010))

print('medians: ', print_round(np.nanmedian(beta_0910_en0505),e05_err), print_round(np.nanmedian(beta_0910_en1010), e10_err)
      , print_round(np.nanmedian(beta_0910_sp0505), s05_err), print_round(np.nanmedian(beta_0910_sp1010), s10_err))

print('errors: ',stats.median_abs_deviation(beta_0910_en0505), stats.median_abs_deviation(beta_0910_en1010)
      , stats.median_abs_deviation(beta_0910_sp0505), stats.median_abs_deviation(beta_0910_sp1010))

print("9602: EN0505    EN1010    SP0505   SP1010")
print('size', len(beta_9602_en0505), len(beta_9602_en1010), len(beta_9602_sp0505), len(beta_9602_sp1010))
print('medians: ', np.nanmedian(beta_9602_en0505), np.nanmedian(beta_9602_en1010), np.nanmedian(beta_9602_sp0505), np.nanmedian(beta_9602_sp1010))
print('errors: ', stats.median_abs_deviation(beta_9602_en0505), stats.median_abs_deviation(beta_9602_en1010), stats.median_abs_deviation(beta_9602_sp0505), stats.median_abs_deviation(beta_9602_sp1010))

# print('9602 and 0910')
# print('medians: ', np.nanmedian(beta_en0505), np.nanmedian(beta_en1010), np.nanmedian(beta_sp0505), np.nanmedian(beta_sp1010))
# print('errors: ', stats.median_abs_deviation(beta_en0505), stats.median_abs_deviation(beta_en1010), stats.median_abs_deviation(beta_sp0505), stats.median_abs_deviation(beta_sp1010))


# beta_0910_en0505 = np.array(df_0910_en0505[['beta1', 'beta2', 'beta3', 'beta4']])
# beta_0910_en1010 = np.array(df_0910_en1010[['beta1', 'beta2', 'beta3', 'beta4']])
# beta_0910_sp0505 = np.array(df_0910_sp0505[['beta1', 'beta2', 'beta3', 'beta4']])
# beta_0910_sp1010 = np.array(df_0910_sp1010[['beta1', 'beta2', 'beta3', 'beta4']])
#
# beta_9602_en0505 = np.array(df_9602_en0505[['beta1', 'beta2']])
# beta_9602_en1010 = np.array(df_9602_en1010[['beta1', 'beta2']])
# beta_9602_sp0505 = np.array(df_9602_sp0505[['beta1', 'beta2']])
# beta_9602_sp1010 = np.array(df_9602_sp1010[['beta1', 'beta2']])
#
# beta_en0505 = np.concatenate([beta_0910_en0505, beta_9602_en0505 ], axis= None)
# beta_en1010 = np.concatenate([beta_0910_en1010, beta_9602_en1010 ], axis= None)
# beta_sp0505 = np.concatenate([beta_0910_sp0505, beta_9602_sp0505 ], axis= None)
# beta_sp1010 = np.concatenate([beta_0910_sp1010, beta_9602_sp1010 ], axis= None)
#
# print('All betas used for each Sim')
# print("0910: EN0505    EN1010    SP0505   SP1010")
# print('medians: ', np.nanmedian(beta_0910_en0505), np.nanmedian(beta_0910_en1010), np.nanmedian(beta_0910_sp0505), np.nanmedian(beta_0910_sp1010))
# print('errors: ', stats.median_abs_deviation(beta_0910_en0505), stats.median_abs_deviation(beta_0910_en1010), stats.median_abs_deviation(beta_0910_sp0505), stats.median_abs_deviation(beta_0910_sp1010))
# print("9602: EN0505    EN1010    SP0505   SP1010")
# print('medians: ', np.nanmedian(beta_9602_en0505), np.nanmedian(beta_9602_en1010), np.nanmedian(beta_9602_sp0505), np.nanmedian(beta_9602_sp1010))
# print('errors: ', stats.median_abs_deviation(beta_9602_en0505), stats.median_abs_deviation(beta_9602_en1010), stats.median_abs_deviation(beta_9602_sp0505), stats.median_abs_deviation(beta_9602_sp1010))
#
# print('9602 and 0910')
# print('medians: ', np.nanmedian(beta_en0505), np.nanmedian(beta_en1010), np.nanmedian(beta_sp0505), np.nanmedian(beta_sp1010))
# print('errors: ', stats.median_abs_deviation(beta_en0505), stats.median_abs_deviation(beta_en1010), stats.median_abs_deviation(beta_sp0505), stats.median_abs_deviation(beta_sp1010))
