
import pandas as pd
import numpy as np


def cuts2017(dfm):
    dfm = dfm.drop(dfm[(dfm.PO3 < 0)].index)
    dfm = dfm.drop(dfm[(dfm.PO3_OPM < 0)].index)

    dfm = dfm.drop(dfm[((dfm.Sim == 175))].index)

    # dfm = dfm.drop(dfm[(dfm.Sim == 179) & (dfm.Team == 4) & (dfm.Tsim > 4000)].index)
    # dfm = dfm.drop(dfm[(dfm.Sim == 172) & (dfm.Tsim < 500)].index)
    # dfm = dfm.drop(dfm[(dfm.Sim == 172) & (dfm.Team == 1) & (dfm.Tsim > 5000) & (dfm.Tsim < 5800)].index)
    # dfm = dfm.drop(dfm[(dfm.Sim == 172) & (dfm.Team == 4) & (dfm.Tsim > 5000)].index)
    #
    # dfm = dfm.drop(dfm[(dfm.Sim == 178) & (dfm.Team == 3) & (dfm.Tsim > 1700) & (dfm.Tsim < 2100)].index)
    # dfm = dfm.drop(dfm[(dfm.Sim == 178) & (dfm.Team == 3) & (dfm.Tsim > 2500) & (dfm.Tsim < 3000)].index)

    # dfm = dfm.drop(dfm[((dfm.Sim == 186) & (dfm.Tsim > 5000))].index)
    dfm = dfm.drop(dfm[((dfm.Sim == 186) & (dfm.Tsim > 4500))].index)

    dfm = dfm.drop(dfm[((dfm.Tsim > 7000))].index)

    return dfm


def cuts0910(dfm):
    dfm = dfm.drop(dfm[(dfm.PO3 < 0)].index)
    dfm = dfm.drop(dfm[(dfm.PO3_OPM < 0)].index)

    dfm = dfm[dfm.ADX == 0]
    # # v2 cuts, use this and v3 standard more conservative cuts not valid for 140, 1122, 163, 166  v2
    dfm = dfm[dfm.Tsim > 900]
    dfm = dfm[dfm.Tsim <= 8100]
    dfm = dfm.drop(dfm[(dfm.Sim == 141) & (dfm.Team == 3)].index)
    # dfm = dfm.drop(dfm[(dfm.Sim == 143) & (dfm.Team == 2) & (dfm.Tsim > 7950) & (dfm.Tsim < 8100)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 147) & (dfm.Team == 3)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 158) & (dfm.Team == 2)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 167) & (dfm.Team == 4)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 158) & (dfm.Team == 1)].index)


    return dfm


def cuts9602(dfm):
    dfm = dfm.drop(dfm[(dfm.Sim == 92) & (dfm.Team == 3)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 98) & (dfm.Team == 7)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 99) & (dfm.Team == 7)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 97) & (dfm.Team == 6)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 98) & (dfm.Team == 6)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 99) & (dfm.Team == 6)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 92) & (dfm.Team == 4)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 97) & (dfm.Team == 5)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 98) & (dfm.Team == 5)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 99) & (dfm.Team == 5)].index)

    return dfm


def cuts0910_beta(dfm):
    dfm = dfm[dfm.ADX == 0]
    # v2 cuts, use this and v3 standard more conservative cuts not valid for 140, 1122, 163, 166  v2
    dfm = dfm.drop(dfm[(dfm.Sim == 141) & (dfm.Team == 3)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 147) & (dfm.Team == 3)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 158) & (dfm.Team == 2)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 158) & (dfm.Team == 1)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 159) & (dfm.Team == 1)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 159) & (dfm.Team == 4)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 167) & (dfm.Team == 4)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 163) & (dfm.Team == 4)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 160) & (dfm.Team == 4)].index)
    dfm = dfm.drop(dfm[(dfm.Sim == 165) & (dfm.Team == 4)].index)

    # dfm = dfm.drop(dfm[(dfm.Sim == '141-3')].index)
    # dfm = dfm.drop(dfm[(dfm.Sim == '147-3')].index)
    # dfm = dfm.drop(dfm[(dfm.Sim == '158-2')].index)
    # dfm = dfm.drop(dfm[(dfm.Sim == '158-1')].index)
    # dfm = dfm.drop(dfm[(dfm.Sim == '159-1')].index)
    # dfm = dfm.drop(dfm[(dfm.Sim == '159-4')].index)
    # dfm = dfm.drop(dfm[(dfm.Sim == '167-4')].index)
    # dfm = dfm.drop(dfm[(dfm.Sim == '163-4')].index)
    # dfm = dfm.drop(dfm[(dfm.Sim == '160-4')].index)
    # dfm = dfm.drop(dfm[(dfm.Sim == '165-4')].index)

    return dfm