
import pandas as pd
import numpy as np
import need_function as nf

# import data
store = pd.HDFStore(r'C:\Users\lenovo\data_201710.hd5')
data_df_fit_name = store['data_df_fit_name']
data_df = store['data_df']
store.close()

data_df_fit_name.columns = ['trade_date', 'symbol']
# 替换之后所需列的属性名
data_df.rename(columns={'S_DQ_ADJCLOSE':'close', 'S_INFO_WINDCODE':'symbol', 'TRADE_DT':'trade_date', 'S_DQ_ADJOPEN':'opening'}, inplace=True)
cols_to_drop = ['S_DQ_VOLUME', 'S_DQ_ADJPRECLOSE', 'S_DQ_ADJHIGH', 'S_DQ_ADJLOW', 'S_DQ_ADJFACTOR','S_DQ_AMOUNT']
data_df.drop(cols_to_drop, axis=1, inplace=True)
# 将两个df中的trade_date symbol 取交集，所取交集记为 新的data_df
data_df = pd.merge(data_df_fit_name, data_df,on=('trade_date','symbol'), how='inner')

# 输入所需的时间段
in_ = input('pleas input your desired MA periods')
if not in_:
    in_ = ' '.join([str(i) for i in [3, 5, 10, 20, 30, 60, 90, 120, 180, 240, 270, 300]])
L = in_.split(' ')
# 输入收益频率
in_ = input('pleas input your desired frequency')
if not in_:
    in_ = 5
frequency = int(in_)

# 构造均线的属性名列表
L = sorted(L)
MA = ['MA' + i for i in L]
L = [int(j) for j in L]


# 计算MA,ret,ret_IC
# DataFrame used for reg
comp_df = nf.cal_MA_ret(data_df, MA, L, frequency).drop(['close', 'opening'], axis=1, level=1)
# ---------------------------------------------------------------回归---------------------------------------------------
# drop useless columns

beta, dic_data_record, rsq_arr = nf.cal_beta(comp_df, MA)
# ---------------------------------------------------------------求因子--------------------------------
# 预测beta值
beta_predict = beta.rolling(window=25).mean()# .dropna(how='all',axis=0)  ##每个日期对应的beta就是第三天的预测值
beta_predict_shift = beta_predict.shift(1)  # 每个日期对应的beta就是第二天的预测值
beta_predict_shift = beta_predict_shift.rolling(window=5).mean().dropna(how='all',axis=0) #使得预测的beta更加平滑，增加相关性，减低换手率

for Time, beta_p in beta_predict_shift.iterrows():
    # print(beta_p)
    # break
    fac = dic_data_record[Time].loc[:, MA]
    fac['const'] = 1.0
    ret_pred = np.dot(fac.values, beta_p.values)
    dic_data_record[Time]['ret_predict'] = ret_pred  # 预测的实际上是Time+1那天的ret

from scipy import stats

ic_dic = {dt: stats.spearmanr(df['ret_predict'], df['ret_IC'])[0]
                 for dt, df in dic_data_record.items() if 'ret_predict' in df.columns}

ic_df = pd.DataFrame(pd.Series(ic_dic).dropna(), columns=['ic'])
ic_df['rolling_mean'] = ic_df.rolling(window=5).mean()
ic_df['cumsum'] = ic_df.ic.cumsum()
print(ic_df.head())






