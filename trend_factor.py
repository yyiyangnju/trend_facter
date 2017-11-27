import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from pandas.core import datetools
import random
import math
from scipy import stats
import need_function as nf

data_df0 = nf.data_read()
data_df = data_df0.set_index(['symbol', 'trade_date'])
data_df = data_df.sort_index(axis=0)  ######importantt!!!


# Define MA periods
in_ = raw_input('pleas input your desired MA periods')
if in_ == "":
    in_ = ' '.join([str(i) for i in [3, 5, 10, 20, 30, 60, 90, 120, 180, 240, 270, 300]])
L = in_.split(' ')

in_ = raw_input('pleas input your desired frequency')
if in_ == "":
    in_ = 5
frequency = int(in_)
#name of attr MA
L = sorted(L)
MA = ['MA' + i for i in L]
L = [int(j) for j in L]

print("Frequency = {}\n"
      "MA = {}\n"
      "L = {}".format(frequency, MA, L))

# cal ma & ret
data_df = nf.ma_cal(data_df, MA)
data_df = nf.ret_cal(data_df, frequency)

# Regression
data_reg = nf.reg_pre(data_df, MA)
gp = data_reg.groupby('trade_date')

## test collinearity
cond_num = gp.apply(nf.test_collinear, MA)
