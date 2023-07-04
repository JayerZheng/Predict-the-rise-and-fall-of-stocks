# import numpy as np
# import pandas as pd
# from pydotplus import graphviz
# from sklearn.tree import DecisionTreeClassifier as DTC
# from sklearn.tree import export_graphviz
# from sklearn.model_selection import train_test_split
# from sklearn import linear_model
# import re
# import matplotlib.pyplot as plt
#
#
# data = pd.read_csv('data.csv', encoding='gbk')  # 打开表格文件
#
# # 替换数据，将数据变为可分类的特征值
# # print(data)
#
# x = data.iloc[:-1, 1:3].astype(int)  # 将数据的特征进行导入
# y = data.iloc[1:, 3].astype(int)  # 将结果进行导入
# # print(y)
#
# x1_train, x1_test, y1_train, y1_test = train_test_split(x, y, train_size=2/3)#The data are divided into train set and test set
# dtc = DTC(criterion='entropy', min_impurity_decrease=0.003, max_depth=6)  # 基于信息熵确立决策树模型
# reg = linear_model.LinearRegression()# Establish a linear regression model
# reg.fit(x, y)
# print(str(dtc))
#
# price = data.iloc[1:, 3].astype(int)
# parameter = data.iloc[:-1, 1:3].astype(int)
# price_pre = reg.predict(parameter)
# print(price_pre)
# # reg_predict = reg.predict(x1_test)
# # print(reg_predict)
# print(reg.score(x1_test, y1_test))
#
# x = pd.DataFrame(x)
#
# plt.plot(data['trade_date'][201:], price_pre[200:], linewidth=1, color="blue", label="MA1")
# plt.plot(data['trade_date'][201:], data.iloc[201:, 3].astype(int), linewidth=1, color="red", label="MA2")
# plt.xticks(range(0, len(data['trade_date'][201:]), 60), rotation=-15, fontsize=10)
# plt.show()
#
# # dot_data = export_graphviz(dtc, out_file=None,
# #                            feature_names=x.columns,
# #                            class_names=['-1', '1'], filled=True,
# #                            rounded=True, special_characters=True)
# #
# # with open(r'tree.dot', 'w', encoding='utf-8') as f:
# #     f.writelines(dot_data)
#
#
# # # 转换消除乱码
# # f = open(r'tree.dot', "r+", encoding="utf-8")
# # with open(r'tree_res.dot', 'w', encoding="utf-8") as f1:
# #     f1.write(re.sub(r'fontname=helvetica', 'fontname="Microsoft YaHei"', f.read()))
#





# import numpy as np
# import pandas as pd
# from sklearn import preprocessing
# import matplotlib.pyplot as plt
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression as LR
# from sklearn.metrics import cohen_kappa_score
#
# data = pd.read_csv('data.csv', encoding='gbk')  # 打开表格文件
# x = data.iloc[:, 1:3].astype(int)  # 将数据的特征进行导入
# y = data.iloc[:, 6].astype(int)  # 将结果进行导入
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)
# # print(x_train.shape[0]) #数据第一维度的数量(行数)，即训练集样本数
# # print(x_test.shape[0])  #数据第一维度的数量(行数)，即测试集样本数
#
# standard_X = preprocessing.StandardScaler()
# x_train = standard_X.fit_transform(x_train)
# x_test = standard_X.fit_transform(x_test)
#
# lr_model = LR(C=1.0, penalty='l2', solver="saga", multi_class="multinomial")
# lr_model.fit(x_train, y_train)
# reg_predict = lr_model.predict(x_test)
# print(lr_model)
# print(lr_model.score(x_test, y_test))
# # print(lr_model.coef_)
# # print(lr_model.intercept_)
#
# y_pred_test = lr_model.predict(x_test)
# acc_test = np.sum(y_pred_test == y_test) / x_test.shape[0] #分类模型在验证集上的准确度
# print('测试集准确度:', acc_test)
# kappa_value = cohen_kappa_score(y_test, y_pred_test)
# print('kappa系数:', kappa_value)
#
# y_pred_train = lr_model.predict(x_train)
# acc_train = np.sum(y_pred_train == y_train) / x_train.shape[0] #分类模型在验证集上的准确度
# print('训练集准确度:', acc_train)
# kappa_value = cohen_kappa_score(y_train, y_pred_train)
# print('kappa系数:', kappa_value)




import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt



data = pd.read_csv('data.csv', encoding='gbk')  # 打开表格文件
# 替换数据，将数据变为可分类的特征值
print(data)

x = data.iloc[30:, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]  # 将数据的特征进行导入
y = data.iloc[30:, 6]  # 将结果进行导入

print(type(x))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
standard_X = preprocessing.StandardScaler()
x = standard_X.fit_transform(x)  # 对特征值数据进行标准化处理
print(x)
pd.DataFrame(x).to_csv('x.csv')

y = np.float32(tf.keras.utils.to_categorical(y, num_classes=2))  # one-hot coding
print(y)
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y, train_size=2/3)#The data are divided into train set and test set

# 构建网络
# relu
# input_x = tf.keras.Input(shape=15, name='input_x')
# firstLyr_output = tf.keras.layers.Dense(24, activation='relu',name='dense_1')(input_x)
# secondLyr_output = tf.keras.layers.Dense(32, activation='relu',name='dense_2')(firstLyr_output)
# thirdLyr_output = tf.keras.layers.Dense(64, activation='relu',name='dense_3')(secondLyr_output)
# output = tf.keras.layers.Dense(2, activation='softmax', name='pred_output')(thirdLyr_output)
# model = tf.keras.Model(inputs=input_x, outputs=output)
# sigmoid
input_x = tf.keras.Input(shape=15, name='input_x')
firstLyr_output = tf.keras.layers.Dense(24, activation='sigmoid',name='dense_1')(input_x)
secondLyr_output = tf.keras.layers.Dense(64, activation='sigmoid',name='dense_2')(firstLyr_output)
thirdLyr_output = tf.keras.layers.Dense(128, activation='sigmoid',name='dense_3')(secondLyr_output)
output = tf.keras.layers.Dense(2, activation='softmax', name='pred_output')(thirdLyr_output)
model = tf.keras.Model(inputs=input_x, outputs=output)

# 训练网络
opt = tf.optimizers.Adam(5e-4)
model.compile(optimizer=opt, loss=tf.losses.categorical_crossentropy,metrics=['accuracy'])
history = model.fit(x=x1_train, y=y1_train, batch_size=32, epochs=400, validation_data=(x1_test, y1_test))
#
# score = model.evaluate(x1_test, y1_test)
pre = model.predict(x1_test, verbose=0)
# print(pre)
res = []
for i in pre:
    if i[0] >= i[1]:
        res.append(0)
    else:
        res.append(1)
print(res)
print(history.history['accuracy'])
plt.plot(history.history['accuracy'], color='red', label='train loss')
plt.plot(history.history['val_accuracy'], color='blue', label='valid loss')
plt.ylim([0.5, 1.0])
plt.show()

# print(y1_test[:10])
# print(pre[:])
# print(y1_test)
# print("The final score is ", score)

# 加上移动平均
