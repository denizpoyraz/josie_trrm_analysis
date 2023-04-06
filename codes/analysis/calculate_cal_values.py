import numpy as np
import pandas as pd
from scipy import stats
# Libraries
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from data_cuts import cuts0910, cuts2017, cuts9602
from plotting_functions import filter_rdif, filter_rdif_all
from analyse_functions import Calc_average_Dif_yref,apply_calibration, cal_dif
from constant_variables import *
import warnings
warnings.filterwarnings("ignore")



def calculate_intercept(av, bv, p1, p2):

    v_p1 = av + bv * np.log10(p1)
    v_p2 = av + bv * np.log10(p2)

    return v_p1, v_p2

# slist = [0,1,3,4]
# for k in slist:
#     # print('0910', a)
#     # v_1000, v_10 = calculate_intercept(a_0910[k], b_0910[k], 1000, 10)
#     # print('0910', labellist[k], v_1000, v_10)
#     v_1000, v_10 = calculate_intercept(a[k], b[k], 1000, 10)
#     print('0910/2017', labellist[k], v_1000, v_10)

slist = [0,2,4,5]
for k in slist:
    print('2017', labellist[k], a_2017[k], b_2017[k])
    # v_1000, v_10 = calculate_intercept(a_2017[k], b_2017[k], 1000, 10)
    # # print('2017', labellist[k], v_1000, v_10)
    # v_1000, v_10 = calculate_intercept(a[k], b[k], 1000, 10)
    # print('0910/2017', labellist[k], v_1000, v_10)