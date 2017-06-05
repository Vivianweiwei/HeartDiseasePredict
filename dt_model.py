#!/usr/bin/env python
# -- coding: UTF-8 --
import pandas as pd
from random import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

inputfile = './ok_data.csv'
data = pd.read_csv(inputfile)
data = data.as_matrix()
shuffle(data)

p = 0.8
train = data[:int(len(data) * p), :]
test = data[int(len(data) * p):, :]

treefile = './tree.pkl'
tree = DecisionTreeClassifier()
tree.fit(train[:, :13], train[:, 13])

joblib.dump(tree, treefile)

# print tree.predict(test[:, :13])

cm = confusion_matrix(train[:, 13], tree.predict(train[:, :13]))
plt.matshow(cm, cmap=plt.cm.Greens)
plt.colorbar()

for x in range(len(cm)):  # 数据标签
    for y in range(len(cm)):
        plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

plt.ylabel('True label')  # 坐标轴标签
plt.xlabel('Predicted label')  # 坐标轴标签
plt.show()  # 显示作图结果

from sklearn.metrics import roc_curve  # 导入ROC曲线函数

fpr, tpr, thresholds = roc_curve(test[:, 13], tree.predict_proba(test[:, :13])[:, 1], pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label='ROC of CART', color='green')  # 作出ROC曲线
plt.xlabel('False Positive Rate')  # 坐标轴标签
plt.ylabel('True Positive Rate')  # 坐标轴标签
plt.ylim(0, 1.05)  # 边界范围
plt.xlim(0, 1.05)  # 边界范围
plt.legend(loc=4)  # 图例
plt.show()  # 显示作图结果
