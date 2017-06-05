import sys
from PyQt5 import QtWidgets
from heart_api import Heart
import numpy as np
from caffe import Ui_MainWindow  # caffe是转换后生成的.py文件


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)  # 创建主界面对象
        Ui_MainWindow.__init__(self)  # 主界面对象初始化
        self.setupUi(self)  # 配置主界面对象

    def getText(self):
        netfile = './net.model'  # 模型路径
        heart = Heart(netfile)
        age = float(self.lineEdit.text())
        sex = float(self.lineEdit_2.text())
        cp = float(self.lineEdit_3.text())
        tresbps = float(self.lineEdit_4.text())
        chol = float(self.lineEdit_5.text())
        fbs = float(self.lineEdit_6.text())
        restecg = float(self.lineEdit_7.text())
        thalach = float(self.lineEdit_8.text())
        exang = float(self.lineEdit_9.text())
        oldpeak = float(self.lineEdit_10.text())
        slope = float(self.lineEdit_11.text())
        ca = float(self.lineEdit_12.text())
        thal = float(self.lineEdit_13.text())
        data = [[age, sex, cp, tresbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        result=heart.predict(np.matrix(data))
        # result={'sickRat': [[ 0.00100181]], 'isSick':[[0]]}
        isSick=result['isSick'][0][0]
        sickRat=result['sickRat'][0][0]
        self.lineEdit_15.setText(str(isSick))
        self.lineEdit_16.setText(str(sickRat))




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
