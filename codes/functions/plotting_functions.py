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
