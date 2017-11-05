
###################################################33 每一天的回归函数
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter


def regress_one_row(ser, MA):

    # argu:data for daily regression
    time = ser.name

    # 生成回归所需数据
    df = ser.unstack(level=1)

    # 除去空值
    df1 = df.dropna(how='any')

    # 字典更新
    dic_data_record[time] = df1

    y_df = df1.loc[:, 'ret']
    x_df = df1.loc[:, MA]

    if x_df.empty or y_df.empty:
        return None

    # 下面进行回归
    LR = LinearRegression()
    LR.fit(x_df, y_df)
    res = np.insert(LR.coef_, len(MA), LR.intercept_)
    rsq_arr.append(LR.score(x_df, y_df))
    return res


# 循环回归求得beta


def cal_beta(comp_df_droped_reg, MA):

    # argu :comp_df_droped_reg 为用于回归的dataframe，包含了所需的完整的数据
    #       MA 用于表示columns
    # return:beta
    from collections import OrderedDict
    reg_result = OrderedDict()
    global dic_data_record
    global rsq_arr
    dic_data_record = OrderedDict()
    rsq_arr = []  # record regression score
    for Time, ser in comp_df_droped_reg.iterrows():
        beta0 = regress_one_row(ser, MA)
        if beta0 is None:
            continue
        reg_result[Time] = beta0  # reg_result 的 values 为 np.array
    beta_arr = np.vstack(reg_result.values())  # beta的雏形：叠在一起的array
    cols = ['beta_' + s for s in MA + ['const']]
    beta = pd.DataFrame(index=reg_result.keys(), columns=cols, data=beta_arr)  #将雏形赋值给df得到最终的beta
    return beta, dic_data_record, rsq_arr

######################################################## 画图



class MyFormatter(Formatter):
    def __init__(self, dates, fmt='%Y%m'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        """Return the label for time x at position pos"""
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''

        # return self.dates[ind].strftime(self.fmt)
        return pd.to_datetime(self.dates[ind], format="%Y%m%d").strftime(self.fmt)


def plot_ic(ic_df):

    ic_arr = ic_df['ic'].values
    ic_mean, ic_std = np.mean(ic_arr), np.std(ic_arr)

    plt.figure(figsize=(16, 8), dpi=300)
    plt.plot(ic_arr, lw=0.5, alpha=0.2, label='Daily IC')
    plt.plot(ic_df['cumsum'].values, lw=1.2, alpha=1, color='red', label='Monthly Rolling Mean')
    plt.axhline(ic_mean, color='k', ls='--', lw=1.2, label='Mean')
    plt.axhline(ic_mean - 3 * ic_std, color='green', ls='--', lw=1.2, label='Mean - 3 Std')
    plt.axhline(ic_mean + 3 * ic_std, color='green', ls='--', lw=1.2, label='Mean + 3 Std')

    plt.gca().xaxis.set_major_formatter(MyFormatter(ic_df.index, '%Y-%m'))

    plt.legend(loc='upper left')
    plt.ylabel('IC Value')
    plt.xlabel('Date')
    plt.title("IC Time Series For Factor {}. Mean = {:.1f}%, Std = {:.1f}%".format(MA, ic_mean * 100, ic_std * 100))

    plt.savefig('MA_IC.pdf')


def cal_MA_ret(df, MA, L, frequency):

    # 计算MA
    # argu: df 提供原始数据的 dataframe
    #      MA 需要计算的MA(str)
    #      L  需要计算MA的int
    # return:含有MA，ret,ret_IC，并且shift过的透视表
    # 默认计算ret

    df = df.sort_values('trade_date')
    gp = df.groupby('symbol')['close']

    for i, ma_period in enumerate(MA):
        temp = gp.rolling(L[i]).mean()
        temp.index = temp.index.droplevel(level=0)
        # temp.sort_index(inplace=True)
        df.loc[:, ma_period] = temp / df['close']

    gp = df.groupby('symbol')
    df.loc[:, 'ret'] = gp['close'].pct_change(periods=frequency)  # .shift(-frequency)#这个没有shift，但是不明白为什么没有按照分组表示
    df.loc[:, 'ret_IC'] = gp['opening'].pct_change(periods=frequency)

    tmp = df.pivot(index='trade_date', columns='symbol')
    comp = tmp.swaplevel(axis=1).sort_index(axis=1)

    comp.loc[:, (slice(None), ['ret', 'ret_IC'])] = comp.loc[:, (slice(None), ['ret', 'ret_IC'])].shift(-frequency - 1)
    return comp
    # comp_df.loc[:,(slice(None),MA)] = comp_df.loc[:,(slice(None),MA)].shift(50)  ##检验未来函数
