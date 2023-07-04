import pandas as pd
import matplotlib.pyplot as plt
import os
import tushare as ts
plt.rcParams['font.sans-serif'] = ['SimHei']

ts.set_token('e2e54e00d4b2df359e4e0c88c65eeabc1da5e4d5877804239c24efba')
pro = ts.pro_api()


def get_data(code, start, end):
    df = pro.daily(ts_code=code, autype='qfq', start_date=start, end_date=end)
    # print(df)
    df.index = pd.to_datetime(df.trade_date)
    # 设置把日期作为索引
    df['ma'] = 0.0  # Backtrader需要用到
    df['openinterest'] = 0.0  # Backtrader需要用到
    # 定义两个新的列ma和openinterest
    df = df[['open', 'high', 'low', 'close', 'vol']]
    # 重新设置df取值，并返回df
    return df


def acquire_code():  # 只下载一只股票数据，且只用CSV保存   未来可以有自己的数据库
    # inp_code = input("请输入股票代码:\n")
    # inp_start = input("请输入开始时间:\n")
    # inp_end = input("请输入结束时间:\n")
    inp_code = '600519.SH'
    inp_start = "20170103"
    inp_end = "20201231"
    df = get_data(inp_code, inp_start, inp_end)
    df.sort_index(inplace=True)

    # print(df.info())

    # 输出统计各列的数据量
    # print("—"*30)
    # 分割线
    # print(df.describe())
    # 输出常用统计参数
    # 把股票数据按照时间正序排列


    ud = []

    print(df['close'][0])
    for i in range(len(df['close'])-1):
        if df['close'][i] < df['close'][i + 1]:
            ud.append(1)
        else:
            ud.append(0)
    print(ud)
    df.drop(df.index[-1], axis=0, inplace=True)
    df['ud'] = ud
    print(df)

    df.to_csv('data.csv')


    # path = os.path.join(os.path.join(os.getcwd(),
    #     "data"), inp_code + ".csv")
    # # os.path地址拼接，''数据地址''为文件保存路径
    # # path = os.path.join(os.path.join(os.getcwd(),"数据地址"),inp_code+"_30M.csv")
    # df.to_csv(path)



acquire_code()
