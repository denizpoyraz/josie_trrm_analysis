import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from numpy.polynomial import polynomial as P
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

# from functions.josie_functions import Calc_average_profileCurrent_pressure, Calc_average_profile_time, Calc_average_profile_Pair, Calc_average_profile_pressure
# from functions.plotting_functions import  filter_rdif_all
from data_cuts import cuts2017

def filter_solsonde(df):
    filtEN = df.ENSCI == 1
    filtSP = df.ENSCI == 0

    filtS10 = df.Sol == 1
    filtS05 = df.Sol == 0.5

    filtB10 = df.Buf == 1.0
    filtB05 = df.Buf == 0.5

    filterEN0505 = (filtEN & filtS05 & filtB05)
    filterEN1010 = (filtEN & filtS10 & filtB10)

    profEN0505 = df.loc[filterEN0505]
    profEN1010 = df.loc[filterEN1010]
    profEN0505_nodup = profEN0505.drop_duplicates(['Sim', 'Team'])
    profEN1010_nodup = profEN1010.drop_duplicates(['Sim', 'Team'])

    ###
    filterSP1010 = (filtSP & filtS10 & filtB10)
    filterSP0505 = (filtSP & filtS05 & filtB05)

    profSP1010 = df.loc[filterSP1010]
    profSP0505 = df.loc[filterSP0505]
    profSP1010_nodup = profSP1010.drop_duplicates(['Sim', 'Team'])
    profSP0505_nodup = profSP0505.drop_duplicates(['Sim', 'Team'])

    return profEN0505_nodup, profEN1010_nodup, profSP0505_nodup, profSP1010_nodup

def filter_df(df):
    filtEN = df.ENSCI == 1
    filtSP = df.ENSCI == 0

    filtS10 = df.Sol == 1
    filtS05 = df.Sol == 0.5

    filtB10 = df.Buf == 1.0
    filtB05 = df.Buf == 0.5

    filterEN0505 = (filtEN & filtS05 & filtB05)
    filterEN1010 = (filtEN & filtS10 & filtB10)

    profEN0505 = df.loc[filterEN0505]
    profEN1010 = df.loc[filterEN1010]
    profEN0505_nodup = profEN0505.drop_duplicates(['Sim', 'Team'])
    profEN1010_nodup = profEN1010.drop_duplicates(['Sim', 'Team'])

    ###
    filterSP1010 = (filtSP & filtS10 & filtB10)
    filterSP0505 = (filtSP & filtS05 & filtB05)

    profSP1010 = df.loc[filterSP1010]
    profSP0505 = df.loc[filterSP0505]
    profSP1010_nodup = profSP1010.drop_duplicates(['Sim', 'Team'])
    profSP0505_nodup = profSP0505.drop_duplicates(['Sim', 'Team'])

    sim_en0505 = profEN0505_nodup.Sim.tolist()
    team_en0505 = profEN0505_nodup.Team.tolist()

    sim_en1010 = profEN1010_nodup.Sim.tolist()
    team_en1010 = profEN1010_nodup.Team.tolist()

    sim_sp0505 = profSP0505_nodup.Sim.tolist()
    team_sp0505 = profSP0505_nodup.Team.tolist()

    sim_sp1010 = profSP1010_nodup.Sim.tolist()
    team_sp1010 = profSP1010_nodup.Team.tolist()

    sim = [sim_en0505, sim_en1010, sim_sp0505, sim_sp1010]
    team = [team_en0505, team_en1010, team_sp0505, team_sp1010]

    return sim, team


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


def apply_calibration(dft, coef):

    prof = filter_rdif_all(dft)

    for j in range(len(prof)):
        prof[j]['R'] = coef[j][1] + coef[j][0] * np.log10(prof[j]['Pair'])
        prof[j]['I_corrected'] = prof[j]['Ifast_minib0_deconv_sm10'] * (100 / (100 + prof[j]['R']))

    return prof

def cal_dif(df, var1, var2, adif, rdif):

    df[adif] = df[var1] - df[var2]
    df[rdif] = (df[var1] - df[var2])/df[var2] * 100

    return df[adif], df[rdif]

def Calc_average_Dif_yref(dataframelist, xcolumn, opmcolumn,  stringy, yref):

    nd = len(dataframelist)

    ybin = 400
    tmin = 200
    tmax = 8000
    ybin0 = ybin
    ymax = tmax
    fac = 1.0
    ystart = tmin

    if stringy =='pressure': n = len(yref) - 1
    if stringy == 'time':     n = math.floor(ymax / ybin0)


    Ygrid = [-9999.0] * n

    Xgrid = [[-9999.0] * n for i in range(nd)]
    OPMgrid = [[-9999.0] * n for i in range(nd)]
    Xsigma = [[-9999.0] * n for i in range(nd)]

    Agrid = [[-9999.0] * n for i in range(nd)]
    Asigma = [[-9999.0] * n for i in range(nd)]

    for j in range(nd):
        dft = dataframelist[j]
        # dft.PFcor = dft.xcolumn

        for i in range(n):
            dftmp1 = pd.DataFrame()
            dfgrid = pd.DataFrame()

            if stringy == 'pressure':
                grid_min = yref[i + 1]
                grid_max = yref[i]
                Ygrid[i] = (grid_min + grid_max) / 2.0
                filta = dft.Pair >= grid_min
                filtb = dft.Pair < grid_max

            if stringy == 'time':

                grid_min = ystart + fac * float(ybin0) * float(i)
                grid_max = ystart + fac * float(ybin0) * float(i + 1)
                Ygrid[i] = (grid_min + grid_max) / 2.0
                filta = dft.Tsim >= grid_min
                filtb = dft.Tsim < grid_max


            filter1 = filta & filtb
            dftmp1['X'] = dft[filter1][xcolumn]
            dftmp1[opmcolumn] = dft[filter1][opmcolumn]

            filtnull = dftmp1.X > -9999.0
            dfgrid['X'] = dftmp1[filtnull].X
            dfgrid[opmcolumn] = dftmp1[filtnull][opmcolumn]

            Xgrid[j][i] = np.nanmean(dfgrid.X)
            Xsigma[j][i] = np.nanstd(dfgrid.X)
            OPMgrid[j][i] = np.nanmean(dfgrid[opmcolumn])

            Agrid[j][i] = np.nanmean(dfgrid.X - dfgrid[opmcolumn])
            Asigma[j][i] = np.nanstd(dfgrid.X - dfgrid[opmcolumn])


    dimension = len(Ygrid)
    nol = len(Xgrid)

    A1verr = [[-9999.0] * dimension for i in range(nol)]
    A1v = [[-9999.0] * dimension for i in range(nol)]
    R1verr = [[-9999.0] * dimension for i in range(nol)]
    R1v = [[-9999.0] * dimension for i in range(nol)]

    for k in range(nol):
        profO3X = Xgrid[k]
        profOPMX = OPMgrid[k]
        profO3Xerr = Asigma[k]
        for ik in range(dimension):
            A1v[k][ik] = (profO3X[ik] - profOPMX[ik])
            A1verr[k][ik] = (profO3Xerr[ik])
            R1v[k][ik] = 100 * (profO3X[ik] - profOPMX[ik]) / profOPMX[ik]
            R1verr[k][ik] = 100 * (profO3Xerr[ik] / profOPMX[ik])

    return  A1v, A1verr, R1v, R1verr, Ygrid




def Calc_average_profile_pressure(dataframelist, xcolumn):
    nd = len(dataframelist)

    yref = [1000, 850, 700, 550, 400, 350, 300, 200, 150, 100, 75, 50, 35, 25, 20, 15,
            12, 10, 8, 6]

    # yref = [1000, 850, 750, 650,  550, 450, 350, 300, 200, 175, 150, 125, 100, 80, 60, 50, 40, 35, 30, 25, 20, 15,
    #         12, 10, 8, 6]
    # #
    # yref = [1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 325, 300, 275, 250, 225, 200, 175,
    #         150, 135, 120, 105, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 28, 26, 24, 22, 20, 18, 16, 14,
    #         12, 10, 8, 6]

    n = len(yref) - 1
    Ygrid = [-9999.0] * n

    Xgrid = [[-9999.0] * n for i in range(nd)]
    Xsigma = [[-9999.0] * n for i in range(nd)]

    Agrid = [[-9999.0] * n for i in range(nd)]
    Asigma = [[-9999.0] * n for i in range(nd)]

    for j in range(nd):
        dft = dataframelist[j]
        dft.PFcor = dft[xcolumn]

        for i in range(n):
            dftmp1 = pd.DataFrame()
            dfgrid = pd.DataFrame()

            grid_min = yref[i + 1]
            grid_max = yref[i]
            Ygrid[i] = (grid_min + grid_max) / 2.0

            filta = dft.Pair >= grid_min
            filtb = dft.Pair < grid_max
            filter1 = filta & filtb
            dftmp1['X'] = dft[filter1].PFcor
            dftmp1['PO3_OPM'] = dft[filter1]['PO3_OPM']

            filtnull = dftmp1.X > -9999.0
            dfgrid['X'] = dftmp1[filtnull].X
            dfgrid['PO3_OPM'] = dftmp1[filtnull].PO3_OPM

            Xgrid[j][i] = np.nanmean(dfgrid.X)
            Xsigma[j][i] = np.nanstd(dfgrid.X)

            Agrid[j][i] = np.nanmean(dfgrid.X - dfgrid['PO3_OPM'])
            Asigma[j][i] = np.nanstd(dfgrid.X - dfgrid['PO3_OPM'])

            # print('j', j, 'i',i, Xgrid[j][i])

    return Xgrid, Xsigma, Ygrid


def polyfit(dfp):

    dfp = cuts2017(dfp)

    dfp['TPintC'] = dfp['TPext'] - 273
    dfp['TPextC'] = dfp['TPint'] - 273
    # dfp['TPintK'] = dfp['TPext']
    # dfp['TPextK'] = dfp['TPint']

    dfp = dfp[dfp.Sim > 185]

    dfen = dfp[dfp.ENSCI == 1]
    dfsp = dfp[dfp.ENSCI == 0]

    avgprof_tpint_en, avgprof_tpint_en_err, Y = Calc_average_profile_pressure([dfen], 'TPintC')
    avgprof_tpext_en, avgprof_tpext_en_err, Y = Calc_average_profile_pressure([dfen], 'TPextC')

    avgprof_tpint_sp, avgprof_tpint_sp_err, Y = Calc_average_profile_pressure([dfsp], 'TPintC')
    avgprof_tpext_sp, avgprof_tpext_sp_err, Y = Calc_average_profile_pressure([dfsp], 'TPextC')

    adifall_en = [i - j for i, j in zip(avgprof_tpint_en[0], avgprof_tpext_en[0])]
    adifall_en_err = [np.sqrt(i * i + j * j) for i, j in zip(avgprof_tpint_en_err[0], avgprof_tpext_en_err[0])]

    adifall_sp = [i - j for i, j in zip(avgprof_tpint_sp[0], avgprof_tpext_sp[0])]
    adifall_sp_err = [np.sqrt(i * i + j * j) for i, j in zip(avgprof_tpint_sp_err[0], avgprof_tpext_sp_err[0])]

    p_en = np.poly1d(np.polyfit(Y, adifall_en, 15))
    p_sp = np.poly1d(np.polyfit(Y, adifall_sp, 15))

    # print('Y', Y)
    print('p_en', p_en)

    return p_en, p_sp

def VecInterpolate_log(XValues, YValues, unc_YValues, dft, Pair):
    # dft = dft.reset_index()
    dft['Cpf'] = 1
    dft['unc_Cpf'] = 1
    dft['plog'] = np.log10(dft[Pair])

    for k in range(len(dft)):
        dft.at[k, 'Cpf'] = 1

        for i in range(len(XValues) - 1):
            # check that value is in between xvalues
            if (XValues[i] >= dft.at[k, 'plog'] >= XValues[i + 1]):
                x1 = float(XValues[i])
                x2 = float(XValues[i + 1])
                y1 = float(YValues[i])
                y2 = float(YValues[i + 1])
                unc_y1 = float(unc_YValues[i])
                unc_y2 = float(unc_YValues[i + 1])
                dft.at[k, 'Cpf'] = y1 + (dft.at[k, 'plog'] - x1) * (y2 - y1) / (x2 - x1)
                # if k > 500:
                # print('Cpf in function',k,  dft.loc[k,'Cpf'])
                dft.at[k, 'unc_Cpf'] = unc_y1 + (dft.at[k, 'plog'] - x1) * (unc_y2 - unc_y1) / (x2 - x1)

    return dft['Cpf'], dft['unc_Cpf']
    # return dft




def filter20(df):
    filtEN = df.ENSCI == 1
    filtSP = df.ENSCI == 0

    filtS20 = df.Sol == 2

    filtB0 = df.Buf == 0.0

    filterEN2000 = (filtEN & filtS20 & filtB0)
    filterSP2000 = (filtSP & filtS20 & filtB0)

    profEN2000 = df.loc[filterEN2000]
    profSP2000 = df.loc[filterSP2000]
    profEN2000_nodup = profEN2000.drop_duplicates(['Sim', 'Team'])
    profSP2000_nodup = profSP2000.drop_duplicates(['Sim', 'Team'])

    sim_en2000 = profEN2000_nodup.Sim.tolist()
    sim_sp2000 = profSP2000_nodup.Sim.tolist()
    team_en2000 = profEN2000_nodup.Team.tolist()
    team_sp2000 = profSP2000_nodup.Team.tolist()



    sim = [sim_en2000, sim_sp2000]
    team = [team_en2000, team_sp2000]

    return sim, team