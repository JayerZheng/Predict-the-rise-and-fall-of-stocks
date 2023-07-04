import copy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 读取文件
data = pd.read_csv('data.csv', encoding='gbk')


def ma(data, N1, N2, N3):
   man1 = data['close'].rolling(N1).mean()
   man2 = data['close'].rolling(N2).mean()
   man3 = data['close'].rolling(N3).mean()
   return man1, man2, man3


# 计算指数平滑移动平均线EMA
def MACD(data):
    EMA12 = pd.DataFrame.ewm(data['close'], span=12).mean()
    EMA26 = pd.DataFrame.ewm(data['close'], span=26).mean()
    DIF = EMA12 - EMA26
    DEA = np.zeros((len(DIF)))
    MACD = np.zeros((len(DIF)))
    for t in range(len(DIF)):
        if t == 0:
            DEA[t] = DIF[t]
        if t > 0:
            DEA[t] = (2*DIF[t]+8*DEA[t-1])/10
        MACD[t] = 2*(DIF[t]-DEA[t])
    return MACD, DIF, DEA


#计算随机指标KDJ
def KDJ(data,N):
    import numpy as np
    Lmin= data['low'].rolling(N).min()
    Lmax= data['high'].rolling(N).max()
    RSV=(data['close'].values-Lmin)/(Lmax-Lmin)
    K=np.zeros((len(RSV)))
    D=np.zeros((len(RSV)))
    J=np.zeros((len(RSV)))
    for t in range(N,len(data)):
        if t==0:
            K[t]=RSV[t]
            D[t]=RSV[t]
        if t>0:
            K[t]=2/3*K[t-1]+1/3*RSV[t]
            D[t]=2/3*D[t-1]+1/3*K[t]
        J[t]=3*D[t]-2*K[t]
    return (K,D,J)


# 计算相对强弱指标RSI
def RSI(data, N):
    import numpy as np
    # z=np.zeros(len(data)-1)
    # z[data.iloc[1:, 6].values-data.iloc[0:-1,2].values>=0]=1
    # z[data.iloc[1:,2].values-data.iloc[0:-1,2].values<0]=-1
    z = copy.deepcopy(data['ud'])
    # z = [z == 1]
    # print(z)
    z1 = pd.DataFrame(z).rolling(N).sum()
    # print(z1)
    print(z)
    z[z == 1] = -1
    z[z == 0] = 1
    z2 = pd.DataFrame(z == 1).rolling(N).sum()
    # z1 = pd.DataFrame(z).rolling(N)[z == 1]
    # print(type(z1)),
    z1 = np.array(z1).ravel()
    z2 = np.array(z2).ravel()
    # print(z2)
    rsi = np.zeros((len(data)))
    for t in range(N, len(data)-1):
        rsi[t] = z1[t]/(z1[t]+z2[t])
    return pd.DataFrame(rsi)


def BIAS(data,N):
    import numpy as np
    bias=np.zeros((len(data)))
    man= data.iloc[:,4].rolling(N).mean()
    for t in range(N-1,len(data)):
        bias[t]=(data.iloc[t,2]-man[t])/man[t]
    return bias


def OBV(data):
    import numpy as np
    obv=np.zeros((len(data)))
    for t in range(len(data)):
        if t==0:
            obv[t]=data['vol'].values[t]
        if t>0:
            if data['close'].values[t]>=data['close'].values[t-1]:
                obv[t]=obv[t-1]+data['vol'].values[t]
            if data['close'].values[t]<data['close'].values[t-1]:
                obv[t]=obv[t-1]-data['vol'].values[t]
    return obv


def cla(data):
    import numpy as np
    y=np.zeros(len(data))
    z=np.zeros(len(y)-1)
    for i in range(len(z)):
        z[data.iloc[1:, 4].values-data.iloc[0:-1, 4].values > 0] = 1
        z[data.iloc[1:, 4].values-data.iloc[0:-1, 4].values == 0] = 0
        z[data.iloc[1:, 4].values-data.iloc[0:-1, 4].values < 0] = -1
        y[i] = z[i]
    return y



# print(RSI(data, 6))



# ma1, ma2, ma3 = ma(data, 10, 30, 120)
# print(len(ma1))
# fig = plt.figure(figsize=(20, 4))
# plt.plot(data['trade_date'][-250:], ma1[-250:], linewidth=1, color="bisque", label="MA1")
# plt.plot(data['trade_date'][-250:], ma2[-250:], linewidth=1, color="orange", label="MA2")
# plt.plot(data['trade_date'][-250:], ma3[-250:], linewidth=1, color="saddlebrown", label="MA3")
# plt.plot(data['trade_date'][-250:], data.iloc[-250:, 4], linewidth=1, color="red", label="Close")
# plt.bar(data['trade_date'][-250:], data.iloc[-250:, 5], linewidth=1, color="blue", label="Vol")
# plt.xticks(range(0, len(data['trade_date'][-250:]), 20), rotation=-15, fontsize=10)
# plt.savefig("demo.pdf", dpi=600, format="pdf")
# plt.legend()
# plt.show()




rsi6 = RSI(data,6)
rsi12 = RSI(data,12)
rsi24 = RSI(data,24)

data['rsi6'] = rsi6
data['rsi12'] = rsi12
data['rsi24'] = rsi24
_ma5, _ma10, _ma20 = ma(data,5,10,20)
data['ma5'] = _ma5
data['ma10'] = _ma10
data['ma20'] = _ma20
macd, dif, dea = MACD(data)
data['macd'] = macd
data['dif'] = dif
_k, _d, _j = KDJ(data, 9)
data['k'] = _k
data['d'] = _d
data['j'] = _j
# fig = plt.figure(figsize=(20, 4))
# plt.plot(data['trade_date'][-250:], _k[-250:], color="yellow", label="k")
# plt.plot(data['trade_date'][-250:], _d[-250:], color="blue", label="d")
# plt.plot(data['trade_date'][-250:], _j[-250:], color="orange", label="j")
# plt.grid(b=True)
# plt.legend()
# plt.xticks(range(0, len(data['trade_date'][-250:]), 60), rotation=-10, fontsize=10)
# plt.show()

bias5 = BIAS(data, 5)
bias10 = BIAS(data, 10)
bias20 = BIAS(data, 20)
# fig = plt.figure(figsize=(20, 4))
# plt.plot(data['trade_date'][:], bias5, color="yellow", label="5")
# plt.plot(data['trade_date'][:], bias10, color="orange", label="10")
# plt.plot(data['trade_date'][:], bias20, color="brown", label="20")
# plt.legend()
# plt.grid(b=True)
# plt.xticks(range(0, len(data['trade_date']), 60), rotation=-10, fontsize=10)
# plt.show()
data['bias5'] = bias5
data['bias10'] = bias10
data['bias20'] = bias20
obv = OBV(data)
data['obv'] = obv
fig = plt.figure(figsize=(40, 8))
# plt.plot(data['trade_date'][:], obv, color="red", label="obv")
plt.plot(data['trade_date'][:], bias5, color="yellow", label="bias5")
plt.plot(data['trade_date'][:], bias10, color="orange", label="bias10")
plt.plot(data['trade_date'][:], bias20, color="brown", label="bias20")
plt.legend()
plt.grid(b=True)
plt.xticks(range(0, len(data['trade_date']), 60), rotation=-10, fontsize=10)
plt.show()
data.to_csv('data.csv', index=False, encoding='gbk')

# fig = plt.figure(figsize=(20, 8))
# sub1 = fig.add_subplot(311)
# plt.plot(data['trade_date'][-500:], data.iloc[-500:, 4], color="blue", label="Vol")
# plt.xticks(range(0, len(data['trade_date'][-500:]), 40), rotation=-10, fontsize=10)
# plt.ylabel(u'price')
#
# sub2 = fig.add_subplot(312)
# plt.bar(data['trade_date'][-500:], macd[-500:], color="red", label="Vol")
# plt.xticks(range(0, len(data['trade_date'][-500:]), 40), rotation=-10, fontsize=10)
# plt.grid(b=True)
# plt.ylabel(u'macd')
#
# sub2 = fig.add_subplot(313)
# plt.plot(data['trade_date'][-500:], dif[-500:], color="yellow", label=r"$y=dif$")
# plt.plot(data['trade_date'][-500:], dea[-500:], linestyle='--', color='brown', label=r'$y=dea$')
# plt.xticks(range(0, len(data['trade_date'][-500:]), 40), rotation=-10, fontsize=10)
# plt.legend()
# plt.savefig("demo_MACD.pdf", dpi=600, format="pdf")
# plt.grid(b=True)
#
# plt.show()
