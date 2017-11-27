import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import statsmodels.api as sm

def data_read(start_date=20060101, end_date=20170928):
    """Only for data from WIND
    return : raw data"""

    data_df_fit_name = pd.read_csv('ashareeodprice20060101-20170928_v_full_members.csv')
    data_df = pd.read_csv('ashareeodprice20060101-20170928.csv')

     # store data
    store = pd.HDFStore('data_201710_bliu.hd5')
    store['data_df_fit_name'] = data_df_fit_name
    store['data_df'] = data_df
    store.close()

     # read data
    #store = pd.HDFStore('data_201710_bliu.hd5')
    #data_df_fit_name = store['data_df_fit_name']
    #data_df = store['data_df']
    #store.close()

    # clean data
    data_df_fit_name.columns = ['trade_date', 'symbol']

    # changed columns' name
    data_df.rename(columns={'S_DQ_ADJCLOSE': 'close', 'S_INFO_WINDCODE': 'symbol', 'TRADE_DT': 'trade_date',
                            'S_DQ_ADJOPEN': 'open'}, inplace=True)
    cols_to_drop = ['S_DQ_VOLUME', 'S_DQ_ADJPRECLOSE', 'S_DQ_ADJHIGH', 'S_DQ_ADJLOW', 'S_DQ_ADJFACTOR', 'S_DQ_AMOUNT']
    data_df.drop(cols_to_drop, axis=1, inplace=True)

    # merge the two dataframe
    data_df = pd.merge(data_df_fit_name, data_df, on=('trade_date', 'symbol'), how='inner')
    return data_df

#---------MA------------------------
def roll_mean_price(df, ma_length=-1):
    return df.rolling(window=ma_length).mean() / df


def ma_cal(df, MA):
    gpb = df.groupby('symbol')['close']
    for i, ma_period in enumerate(MA):
        ser_ma = gpb.apply(roll_mean_price, ma_length=int(ma_period[2:]))  ############
        df.loc[:, ma_period] = ser_ma
        print("{} calculation finished.".format(ma_period))
    return df

# -----------------return-------------
def ret_shift(df, period=1):
    return df.pct_change(periods=period).shift(-period)


def ret_cal(df, frequency):
    gpb = df.groupby('symbol')['close']
    df.loc[:, 'ret'] = gpb.apply(ret_shift, period=frequency)
    # data_df.loc[:,'ret_IC'] = gp['opening'].pct_change(periods=frequency)
    return df

# -------------------regression data-----
def reg_pre(df, MA):
    cols_to_drop = list(set(df.columns) - set(MA + ['ret']))
    data_reg = df.drop(cols_to_drop, axis=1)  # dataframe used for regression
    print data_reg.shape

    mask_to_drop = np.any(data_reg.isnull(), axis=1)
    print("{:.1f}% of data are droped because of all NaN.".format(mask_to_drop.sum() * 100. / mask_to_drop.shape[0]))
    data_reg = data_reg.loc[~mask_to_drop]
    print data_reg.shape
    return data_reg


# -------------------collinearity----------
def test_collinear(df, col):
    return np.linalg.cond(df.loc[:,col])

# -------------------PCA------------------
def unify(df):
    arr = df.values
    return arr / np.linalg.norm(arr)


def PCA_var_reg(df, ma_cols_names=MA, n=1, reg=False):
    """
    ma_cols_names: attributes
    n: components
    if reg == False then we just analyze
    else we do the regression based on the result of decomposition
    """

    x = df.loc[:, ma_cols_names]
    pca = PCA(n_components=n)
    pca.fit(x)
    if reg == False:
        return pca.explained_variance_ratio_
    else:
        xr = pca.fit_transform(x)
        yr = unify(df.ret)
        # Let's regress
        xr = sm.add_constant(xr)
        model = sm.OLS(yr, xr)
        fit_res = model.fit()
        ser = fit_res.params.copy()  # coefficient
        return ser,df
