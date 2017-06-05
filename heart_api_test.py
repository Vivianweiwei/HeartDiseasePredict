#!/usr/bin/env python
# -- coding: UTF-8 --
from heart_api import Heart
import numpy as np

if __name__ == '__main__':
    netfile = './net.model'  # 模型路径
    heart = Heart(netfile)
    data = [[38, 1, 3, 138, 175, 0, 0, 173, 0, 0, 1, 1, 3]]
    result=heart.predict(np.matrix(data))
    # dic={'isSick': array([[0]]), 'sickRat': array([[ 0.00100181]], dtype=float32)}
    print(result)
    print(result['isSick'][0][0])
    print(result['sickRat'][0][0])