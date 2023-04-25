from matplotlib.ticker import ScalarFormatter, MultipleLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from re import search

# from functions.josie_plotfunctions import errorPlot_ARDif_withtext, errorPlot_general


def filter_rdif(dft, bool_9602, bool_0910, bool_2017):
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

    profEN0505_nodup = profEN0505.drop_duplicates(['Sim', 'Team'])
    profEN1010_nodup = profEN1010.drop_duplicates(['Sim', 'Team'])

    print(profEN0505_nodup[['Sim', 'Team']])

    if not bool_9602:
        # totO3_EN0505 = profEN0505_nodup.frac.mean()
        # totO3_EN1010 = profEN1010_nodup.frac.mean()
        totO3_EN0505 = 1
        totO3_EN1010 = 1
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

    profSP1010_nodup = profSP1010.drop_duplicates(['Sim', 'Team'])
    profSP0505_nodup = profSP0505.drop_duplicates(['Sim', 'Team'])

    if not bool_9602:
        # totO3_SP1010 = profSP1010_nodup.frac.mean()
        # totO3_SP0505 = profSP0505_nodup.frac.mean()
        totO3_SP1010 = 1
        totO3_SP0505 = 1
    if bool_9602:
        totO3_SP1010 = 1
        totO3_SP0505 = 1

    prof = [profEN0505, profEN1010, profSP0505, profSP1010]

    return prof


def filter_rdif_all(dft):

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
    profEN1001 = dft.loc[filterEN1001]

    profEN0505_nodup = profEN0505.drop_duplicates(['Sim', 'Team'])
    profEN1010_nodup = profEN1010.drop_duplicates(['Sim', 'Team'])
    profEN1001_nodup = profEN1001.drop_duplicates(['Sim', 'Team'])

    print('profEN0505', len(profEN0505_nodup))
    print('profEN1010', len(profEN1010_nodup))
    print('profEN1001', len(profEN1001_nodup))


    filterSP1010 = (filtSP & filtS10 & filtB10)
    filterSP0505 = (filtSP & filtS05 & filtB05)
    filterSP1001 = (filtSP & filtS10 & filtB01)

    profSP1010 = dft.loc[filterSP1010]
    profSP0505 = dft.loc[filterSP0505]
    profSP1001 = dft.loc[filterSP1001]

    profSP1010_nodup = profSP1010.drop_duplicates(['Sim', 'Team'])
    profSP0505_nodup = profSP0505.drop_duplicates(['Sim', 'Team'])
    profSP1001_nodup = profSP1001.drop_duplicates(['Sim', 'Team'])
    
    print('profSP0505', len(profSP0505_nodup))
    print('profSP1010', len(profSP1010_nodup))
    print('profSP1001', len(profSP1001_nodup))

    prof = [profEN0505, profEN1010, profEN1001, profSP0505, profSP1010, profSP1001]
    # prof = [profEN0505]


    return prof


def errorPlot_ARDif_withtext(xlist, xerrorlist, Y, xra, yra, maintitle, xtitle, ytitle, labelist,
                          plotname, nstat, logbool, textbool):
    '''

    :param xlist: a list of x values
    :param xerrorlist: a list of x error values
    :param Y:
    :param xra: x range
    :param yra: y range
    :param maintitle:
    :param xtitle:
    :param ytitle:
    :param labelist: a list of the labels
    :param O3valuelist: specific total O3 fractions
    :param dfnoduplist: a list of df_noduplicated(sim), in order to know number of simulations
    :param plotname: how you want to save your plot
    :param path: folder name in Plots folder
    :param logbool: True if you want logarithmic scale
    :param textbool: True if you want an additional text
    :return: just plots
    '''

    d = len(xlist)
    n = [0] * d
    labell = [''] * d
    textl = [''] * d
    tl = [''] * d
    # colorl = ['black', 'magenta','red', 'blue', 'green','orange']
    # colorl = ['black','red', 'blue', 'green','orange']
    #0910
    #en0505, en1010, sp0505, sp1010
    # colorl = ['red','black', 'green', 'blue' ]
    # labellist_title = ['EN-SCI 0.5%', 'EN-SCI 1.0%', 'EN-SCI 0.1%','SPC 0.5%', 'SPC 1.0%', 'SPC 0.1%']
    colorl = ['#e41a1c', '#a65628',  '#4daf4a', '#377eb8']

    #2017
    #en0505, en1001, sp1001, sp1010
    # colorl = ['#e41a1c',  '#dede00', '#377eb8', '#984ea3']



    ## y locations of the extra text which is drawn for the total O3 values
    texty = [0.23, 0.16, 0.09, 0.02]
    size_label = 22
    size_title = 24
    size_text = 18

    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 8))
    # plt.figure(figsize=(12, 8))


    fig.subplots_adjust(bottom=0.17)

    plt.xlim(xra)
    plt.ylim(yra)
    plt.title(maintitle, fontsize=size_title)
    plt.xlabel(xtitle, fontsize=size_label)
    plt.ylabel(ytitle, fontsize=size_label)
    plt.xticks(fontsize=size_text)
    plt.yticks(fontsize=size_text)
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().tick_params(which='major', width=2)
    plt.gca().tick_params(which='minor', width=2)
    # plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
    plt.gca().xaxis.set_tick_params(length=5, which='minor')
    plt.gca().xaxis.set_tick_params(length=10, which='major')
    # plt.gca().yaxis.set_tick_params(length=5, which='minor')
    plt.gca().yaxis.set_tick_params(length=10, which='major')


    # plt.yticks(np.arange(0, 7001, 1000))

    # reference line
    ax.axvline(x=0, color='grey', linestyle='--')
    if logbool: ax.set_yscale('log')

    for i in range(d):
        # n[i] = len(dfnoduplist[i])
        if textbool:
            labell[i] = f'{labelist[i]} (n={nstat[i]})'
            print(i, nstat[i], labell[i])
        if not textbool :
                labell[i] = labelist[i]
        # labell[i] = labelist[i]
        # \
                    # + ' ( n =' + str(n[i]) + ')'
        # textl[i] = 'tot O3 ratio: ' + str(round(O3valuelist[0], 2))
        # print(xerrorlist[i])
        ax.errorbar(xlist[i], Y, xerr=xerrorlist[i], label=labell[i], color=colorl[i], linewidth=2,
                    elinewidth=0.5, capsize=1, capthick=0.5)

        # if textbool: tl[i] = ax.text(0.05, texty[i], textl[i], color=colorl[i], transform=ax.transAxes)


    ax.legend(loc='best', frameon=True, fontsize='x-large')

    plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v3/png/'+ plotname + '.png')
    plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v3/pdf/'+ plotname + '.pdf')
    plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Plots_2023_final/v3/eps/'+ plotname + '.eps')
    # plt.show()


    plt.close()


def errorPlot_ARDif(xlist, xerrorlist, Y, xra, yra, maintitle, xtitle, ytitle, labelist,
                          plotname, path, logbool, nstat, Dif_rangex):
    '''

    :param xlist: a list of x values
    :param xerrorlist: a list of x error values
    :param Y:
    :param xra: x range
    :param yra: y range
    :param maintitle:
    :param xtitle:
    :param ytitle:
    :param labelist: a list of the labels
    :param O3valuelist: specific total O3 fractions
    :param dfnoduplist: a list of df_noduplicated(sim), in order to know number of simulations
    :param plotname: how you want to save your plot
    :param path: folder name in Plots folder
    :param logbool: True if you want logarithmic scale
    :param textbool: True if you want an additional text
    :return: just plots
    '''

    d = len(xlist)
    n = [0] * d
    labell = [''] * d
    textl = [''] * d
    tl = [''] * d
    # colorl = ['black', 'magenta','red', 'blue', 'green','orange']
    # colorl = ['black','red', 'blue', 'green','orange']
    #0910
    #en0505, en1010, sp0505, sp1010
    colorl = ['red','black', 'green', 'blue' ]

    #2017
    #en0505, en1001, sp1001, sp1010
    # colorl = ['red','maroon', 'darkcyan', 'blue']
    # colorl = ['red','maroon', 'tab:orange', 'blue']



    ## y locations of the extra text which is drawn for the total O3 values
    texty = [0.23, 0.16, 0.09, 0.02]

    plt.close('all')
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.17)

    plt.xlim(xra)
    plt.ylim(yra)
    plt.title(maintitle, fontsize=14)
    plt.xlabel(xtitle, fontsize=14)
    plt.ylabel(ytitle, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)

    # plt.yticks(np.arange(0, 7001, 1000))

    # reference line
    ax.axvline(x=0, color='grey', linestyle='--')
    if logbool: ax.set_yscale('log')

    for i in range(d):
        # n[i] = len(dfnoduplist[i])
        if nstat[0] != 999:
            labell[i] = f'{labelist[i]} (n={nstat[i]})'
        if nstat[0] == 999:
            labell[i] = labelist[i]
        # \
                    # + ' ( n =' + str(n[i]) + ')'
        # textl[i] = 'tot O3 ratio: ' + str(round(O3valuelist[0], 2))
        # print(xerrorlist[i])
        ax.errorbar(xlist[i], Y, xerr=xerrorlist[i], label=labell[i], color=colorl[i], linewidth=1, elinewidth=0.5, capsize=1, capthick=0.5)

    ax.text(Dif_rangex, 400, 'R1', style='italic', bbox={'facecolor': 'white', 'alpha': 0.6})
    ax.text(Dif_rangex, 85, 'R2', style='italic', bbox={'facecolor': 'white', 'alpha': 0.6})
    ax.text(Dif_rangex, 17, 'R3', style='italic', bbox={'facecolor': 'white', 'alpha': 0.6})
    # ax.text(Dif_rangex, 5, 'R4', style='italic')
    # ax.text(Dif_rangex, 425, 'R1', style='italic', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

    # R1 475-375
    # R2 108-83
    # R3 20-6
    # R4 5.4

    ax.legend(loc='lower left', frameon=True, fontsize='small')

    # plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Paper_Plots_tfast_upd/Less_cuts_' + plotname + '.png')
    # plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Paper_Plots/' + plotname + '.eps')
    # plt.savefig('/home/poyraden/Analysis/JosieAnalysis/Plots/Paper_Plots/' + plotname + '.pdf')

    plt.show()


    plt.close()

