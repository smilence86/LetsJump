# -*- coding: utf-8 -*-

import random, time as time, datetime, os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import *

# matplotlib.use('qt4agg')

myfont = FontProperties(fname='./SourceHanSerifCN-Light.otf')

def showGraph(filepath):
    arr = np.load(filepath)['array'].tolist()[:1000000]
    print(len(arr))
    # print(arr)
    x = np.arange(len(arr))
    y = arr
    # plt.figure()
    matplotlib.rcParams['axes.unicode_minus']=False  
    plt.title(u'loss损失函数', fontproperties=myfont)
    plt.xlabel('训练次数', fontproperties=myfont)
    plt.ylabel('loss')
    plt.plot(x, y)
    plt.show()

showGraph('./time.npz')


