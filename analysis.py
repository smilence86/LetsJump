# -*- coding: utf-8 -*-

import random, time as time, datetime, shutil, os, cv2
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import *

# matplotlib.use('qt4agg')

myfont = FontProperties(fname='./SourceHanSerifCN-Light.otf')

def showGraph(filepath):
    arr = np.load(filepath)['array'].tolist()[:]
    print(len(arr))
    # for t in arr:
    #     if t > 0.3:
    #         print('{0:.10f}'.format(t))
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


def sortByTime(dirpath):
    a = [s for s in os.listdir(dirpath)
        if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: s.split('_')[0])
    return a

def removeLessFolder(path):
    dirs = os.listdir(path)
    dirs.sort()
    for record in dirs:
        if len(os.listdir(path + record)) < 15 and record != '2018-01-30 13:15:00':
            print('删除目录：' + path + record)
            shutil.rmtree(path + record)


# removeLessFolder('./records/');





