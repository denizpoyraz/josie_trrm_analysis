
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from re import search


from data_cuts import cuts0910_beta, cuts9602
from beta_functions import  filter_solsonde

def add_value_label(x_list, y_list, ann_list, y_off):
    print(len(x_list))
    print(len(y_list))
    s_list = ['']*4
    for k in range(len(y_list)):
        y_list[k] = round(y_list[k], 3)
        s_list[k] = "%.3f" % round(y_list[k], 3)
    # y_list = round(y_list,3)
    for i in range(0, len(x_list)):
        # plt.annotate(y_list[i],(i,y_list[i]),ha="center")
        print(i, y_list[i])
        # plt.annotate(y_list[i], (i + ann_list[i], y_list[i] - y_off),
        #              horizontalalignment='left', size='xx-large', color='k',
        #              weight='semibold', bbox=dict(facecolor='white'))
        plt.annotate(s_list[i], (i + ann_list[i], y_list[i] - y_off),
                 horizontalalignment='left', size='xx-large', color='k',
                 weight='semibold', bbox=dict(facecolor='white'))
bool_groups = True
bool_cuts = False
alist = [0.275, 0.275, 0.275, 0.275]
yoffset = 0
# pname = 'Fig_3_boxplot_noiB0'
pname = 'Fig_3_boxplot'

df1 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Data_nocut_beta_paper.csv")
# df1 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Data_nocut_beta_paper_noib0.csv")


df2 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie9602_Data_withcut_beta.csv", low_memory=False)

df1.loc[df1.Sim==144,'beta4'] = -9
df1.loc[(df1.Sim==166) & (df1.Team==1),'beta2'] = -9

df1 = df1[df1.beta4 > 0]
df1 = df1[df1.beta2 > 0]

if bool_cuts:df1 = cuts0910_beta(df1)
# df2 = cuts9602(df2)
df1 = df1[df1.ADX == 0]
df1 = df1.drop(df1[(df1.Sim == 147) & (df1.Team == 3)].index)
df1 = df1.drop(df1[(df1.Sim == 167) & (df1.Team == 4)].index)
df1 = df1.drop(df1[(df1.Sim == 158) & (df1.Team == 1)].index)
df1 = df1.drop(df1[(df1.Sim == 158) & (df1.Team == 2)].index)

df1 = df1[df1.iB0 >= 0]
df_0910_en0505, df_0910_en1010, df_0910_sp0505, df_0910_sp1010 = filter_solsonde(df1)
df_9602_en0505, df_9602_en1010, df_9602_sp0505, df_9602_sp1010 = filter_solsonde(df2)

# df_en0505 = df_0910_en0505['beta'].append(df_9602_en0505['beta'], ignore_index=True)
# df_en1010 = df_0910_en1010['beta'].append(df_9602_en1010['beta'], ignore_index=True)
# df_sp0505 = df_0910_sp0505['beta'].append(df_9602_sp0505['beta'], ignore_index=True)
# df_sp1010 = df_0910_sp1010['beta'].append(df_9602_sp0505['beta'], ignore_index=True)



##          0910 Plots

x1 = [0] * len(df_0910_en0505) * 4
x1_1 = [1] * len(df_0910_en1010) * 4
x1_2 = [2] * len(df_0910_sp0505) * 4
x1_3 = [3] * len(df_0910_sp1010) * 4
x1.extend(x1_1)
x1.extend(x1_2)
x1.extend(x1_3)

xr = [0] * len(df_0910_en0505)
xr_1 = [1] * len(df_0910_en1010)
xr_2 = [2] * len(df_0910_sp0505)
xr_3 = [3] * len(df_0910_sp1010)
xr.extend(xr_1)
xr.extend(xr_2)
xr.extend(xr_3)

en0910 = np.concatenate(
    (df_0910_en0505[['beta1', 'beta2', 'beta3', 'beta4']].to_numpy().reshape(-1), df_0910_en1010[['beta1', 'beta2', 'beta3', 'beta4']].to_numpy().reshape(-1),
     df_0910_sp0505[['beta1', 'beta2', 'beta3', 'beta4']].to_numpy().reshape(-1),
     df_0910_sp1010[['beta1', 'beta2', 'beta3', 'beta4']].to_numpy().reshape(-1)), axis=None)

en0910_upd = np.concatenate(
    (df_0910_en0505['beta'].tolist(), df_0910_en1010['beta'].tolist(),
     df_0910_sp0505['beta'].tolist(),
     df_0910_sp1010['beta'].tolist()), axis=None)

xlabels = ['EN-SCI SST0.5', 'EN-SCI SST1.0', 'SPC SST0.5', 'SPC SST1.0']

r1 = np.concatenate((df_0910_en0505['beta1'].to_numpy().reshape(-1), df_0910_en1010['beta1'].to_numpy().reshape(-1),
                            df_0910_sp0505['beta1'].to_numpy().reshape(-1), df_0910_sp1010['beta1'].to_numpy().reshape(-1)), axis=None)
r2 = np.concatenate((df_0910_en0505['beta2'].to_numpy().reshape(-1), df_0910_en1010['beta2'].to_numpy().reshape(-1),
                            df_0910_sp0505['beta2'].to_numpy().reshape(-1), df_0910_sp1010['beta2'].to_numpy().reshape(-1)), axis=None)
r3 = np.concatenate((df_0910_en0505['beta3'].to_numpy().reshape(-1), df_0910_en1010['beta3'].to_numpy().reshape(-1),
                            df_0910_sp0505['beta3'].to_numpy().reshape(-1), df_0910_sp1010['beta3'].to_numpy().reshape(-1)), axis=None)
r4 = np.concatenate((df_0910_en0505['beta4'].to_numpy().reshape(-1), df_0910_en1010['beta4'].to_numpy().reshape(-1),
                            df_0910_sp0505['beta4'].to_numpy().reshape(-1), df_0910_sp1010['beta4'].to_numpy().reshape(-1)), axis=None)


colors = ("grey", "blue", "red", 'green')
cbl = ['#e41a1c', '#a65628','#dede00', '#4daf4a', '#377eb8', '#984ea3']
# red, brown, yellow, green, blue, purple

# groups = (r"T$_{sim.} 2100-2400$", r"T$_{sim} 4100-4400$", r"T$_{sim} 6100-6400$", r'T$_{sim} 8100-8400$')
groups = ('RT1', 'RT2', 'RT3', 'RT4')


print(len(en0910), len(x1))



#
# ###updated plot using one beta for each sim
plotname = '0910_beta_boxplot_newmethod'
if bool_groups:plotname = '0910_beta_boxplot_newmethod_groups'
if bool_cuts: plotname = plotname + "_0910cuts"

size_label = 20
size_title = 22
size_tick = 18
fig = plt.figure(figsize=(12, 8))
plt.axis('off')
plt.title('2009/2010 JOSIE', fontsize=size_title)
plt.xticks(fontsize=size_tick)
plt.yticks(fontsize=size_tick)
ax1 = fig.add_subplot(1, 1, 1)
if not bool_groups: ax1.scatter(xr, en0910_upd, alpha=0.8, label='0910 Data')
ss = 200
if bool_groups:
    ax1.scatter(xr, r1, alpha=0.8, c=cbl[2], marker="o", label=groups[0], s=ss)
    ax1.scatter(xr, r2, alpha=0.8, c=cbl[4], marker='v', edgecolors='none', label=groups[1], s=ss)
    ax1.scatter(xr, r3, alpha=0.8, c=cbl[0], marker="<", edgecolors='none', label=groups[2], s=ss)
    ax1.scatter(xr, r4, alpha=0.8, c=cbl[3], marker=">", edgecolors='none', label=groups[3], s=ss)

data_to_plot = [df_0910_en0505['beta'].tolist(),
                 df_0910_en1010['beta'].tolist(),
     df_0910_sp0505['beta'].tolist(),
     df_0910_sp1010['beta'].tolist()]

bp = ax1.boxplot(data_to_plot, positions=[0, 1, 2, 3])
medians = [item.get_ydata()[0] for item in bp['medians']]
print('median',medians)
# medians = [0.018,0.046,0.017,0.049]
add_value_label([0, 1, 2, 3], medians, alist, yoffset)
# ax1.bar_label(medians, label_type='edge')
ax1.set_xticks(np.arange(len(xlabels)))
ax1.set_xticklabels(xlabels, fontsize=size_label)
plt.legend(loc='upper left', frameon=True, fontsize='xx-large', handletextpad=0.1)

plt.ylim(-0.05, 0.15)
ax1.set_ylabel(r'Slow stoichiometry factor S$_S$', fontsize=size_label)
# ax1.set_ylabel(r'Stoichiometry Factor S$_S$ of Slow Reaction Pathway', fontsize=size_label)

plt.yticks(fontsize=size_tick)

plt.savefig(f'/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v3/png/{pname}.png')
# plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v3/eps/Fig_3_boxplot.eps')
# plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v3/pdf/Fig_3_boxplot.pdf')
plt.show()
plt.close()
# #######


#
# plotname = '0910_beta_boxplot_oldmethod'
# if bool_groups:plotname = '0910_beta_boxplot_oldmethod_groups'
# if bool_cuts: plotname = plotname + "_0910cuts"
#
# fig = plt.figure(figsize=(10, 5))
# plt.axis('off')
# plt.title('2009-2010 JOSIE', fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# ax1 = fig.add_subplot(1, 1, 1)
# if bool_groups:
#     ax1.scatter(xr, r1, alpha=0.8, c=colors[0], marker="o", label=groups[0])
#     ax1.scatter(xr, r2, alpha=0.8, c=colors[1], marker='v', edgecolors='none', label=groups[1])
#     ax1.scatter(xr, r3, alpha=0.8, c=colors[2], marker="<", edgecolors='none', label=groups[2])
#     ax1.scatter(xr, r4, alpha=0.8, c=colors[3], marker=">", edgecolors='none', label=groups[3])
# if not bool_groups: ax1.scatter(x1, en0910, alpha=0.8, label='0910 Data')
#
# data_to_plot =  [df_0910_en0505[['beta1', 'beta2', 'beta3', 'beta4']].to_numpy().reshape(-1),
#                  df_0910_en1010[['beta1', 'beta2', 'beta3', 'beta4']].to_numpy().reshape(-1),
#      df_0910_sp0505[['beta1', 'beta2', 'beta3', 'beta4']].to_numpy().reshape(-1),
#      df_0910_sp1010[['beta1', 'beta2', 'beta3', 'beta4']].to_numpy().reshape(-1)]
# # ax1.boxplot(data_to_plot, positions=[0,1,2,3])
# bp = ax1.boxplot(data_to_plot, positions=[0, 1, 2, 3])
# medians = [item.get_ydata()[0] for item in bp['medians']]
# print(medians)
# add_value_label([0, 1, 2, 3], medians)
#
# ax1.set_xticks(np.arange(len(xlabels)))
# ax1.set_xticklabels(xlabels, fontsize=12)
# plt.legend(loc='upper left')
# plt.legend(loc='best')
# plt.ylim(-0.05, 0.4)
# ax1.set_ylabel(r'I$_{ECC}$ - iB0 / I$_{OPM}$, conv($\tau_{slow})}$', fontsize=14)
#
# plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Paper_Plots_122022/beta_0512_' + plotname + '.png')
# plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Paper_Plots_122022/beta_0512_' + plotname + '.pdf')
# plt.show()
# plt.close()
#