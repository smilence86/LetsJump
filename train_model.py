# -*- coding: utf-8 -*-

import random, time as time, datetime, shutil, os, cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def weight_variable(shape, std):
    initial = tf.truncated_normal(shape, stddev=std, mean=0)
    return tf.Variable(initial)


def bias_variable(shape, std):
    initial = tf.constant(std, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')


# def tf_ops():
# 输入：100*100的灰度图片，前面的None是batch size，这里都为1
x = tf.placeholder(tf.float32, shape=[None, 100, 100, 1])
# 输出：一个浮点数，就是按压时间，单位s
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# 第一层卷积 12个feature map
W_conv1 = weight_variable([5, 5, 1, 12], 0.1)
b_conv1 = bias_variable([12], 0.1)
# 卷积后为96*96*12

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_4x4(h_conv1)
# 池化后为24*24*12

# 第二层卷积 24个feature map
W_conv2 = weight_variable([5, 5, 12, 24], 0.1)
b_conv2 = bias_variable([24], 0.1)
# 卷积后为20*20*24

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_4x4(h_conv2)
# 池化后为5*5*24

# 全连接层5*5*24 --> 32
W_fc1 = weight_variable([5 * 5 * 24, 32], 0.1)
b_fc1 = bias_variable([32], 0.1)
h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 24])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 全连接层5*5*24 --> 32
W_fc2 = weight_variable([32, 600], 0.1)
b_fc2 = bias_variable([600], 0.1)
# h_pool3_flat = tf.reshape(h_pool3, [-1, 5 * 5 * 24])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

# drapout，play时为1训练时为0.6
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc2, keep_prob)
# 学习率
learn_rate = tf.placeholder(tf.float32)

# 32 --> 1
W_fc3 = weight_variable([600, 1], 0.1)
b_fc3 = bias_variable([1], 0.1)
y_fc3 = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

# 因输出直接是时间值，而不是分类概率，所以用平方损失
cross_entropy = tf.reduce_mean(tf.square(y_fc3 - y_))
train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

tf_init = tf.global_variables_initializer()

saver_init = tf.train.Saver({"W_conv1": W_conv1, "b_conv1": b_conv1,
                             "W_conv2": W_conv2, "b_conv2": b_conv2,
                             "W_fc1": W_fc1, "b_fc1": b_fc1,
                             "W_fc2": W_fc2, "b_fc2": b_fc2,
                             "W_fc3": W_fc3, "b_fc3": b_fc3})
    # return tf_init, saver_init, y_fc3, cross_entropy, train_step, x, y_, keep_prob, learn_rate


# 获取屏幕截图并转换为模型的输入
def get_screen_shot():
    # 使用adb命令截图并获取图片，这里如果把后缀改成jpg会导致TensorFlow读不出来
    os.system('adb shell screencap -p /sdcard/jump_temp.png')
    os.system('adb pull /sdcard/jump_temp.png .')
    # 使用PIL处理图片，并转为jpg
    im = Image.open(r"./jump_temp.png")
    w, h = im.size
    # 将图片压缩，并截取中间部分，截取后为100*100
    im = im.resize((108, 192), Image.ANTIALIAS)
    region = (4, 50, 104, 150)
    im = im.crop(region)
    # 转换为jpg
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, im)
    bg.save(r"./jump_temp.jpg")

    img_data = tf.image.decode_jpeg(tf.gfile.FastGFile('./jump_temp.jpg', 'rb').read())
    # 使用TensorFlow转为只有1通道的灰度图
    img_data_gray = tf.image.rgb_to_grayscale(img_data)
    x_in = np.asarray(img_data_gray.eval(), dtype='float32')

    # [0,255]转为[0,1]浮点
    for i in range(len(x_in)):
        for j in range(len(x_in[i])):
            x_in[i][j][0] /= 255

    # 因为输入shape有batch维度，所以还要套一层
    return [x_in]


# 按压press_time时间后松开，完成一次跳跃
def jump(press_time):

    xs = [random.uniform(950, 980) for _ in range(2)]
    ys = [random.uniform(1750, 1820) for _ in range(2)]
    
    cmd = 'adb shell input swipe {} {} {} {} {}'.format(
        xs[0],
        ys[0], 
        xs[1], 
        ys[1], 
        press_time
    )
    print(cmd)
    os.system(cmd)


# 判断是否游戏失败到分数页面
def has_die(x_in):
    # 判断左上右上左下右下四个点的亮度
    if (x_in[0][0][0][0] < 0.4) and (x_in[0][0][len(x_in[0][0]) - 1][0] < 0.4) and (
        x_in[0][len(x_in[0]) - 1][0][0] < 0.4) and (x_in[0][len(x_in[0]) - 1][len(x_in[0][0]) - 1][0] < 0.4):
        return True
    else:
        return False


# 游戏失败后重新开始，(540，1588)为1080*1920分辨率手机上重新开始按钮的位置
def restart():
    #adb shell wm size
    cmd = 'adb shell input swipe 540 1588 540 1588 {}'.format(int(random.uniform(35, 98)))
    os.system(cmd)
    time.sleep(1)


# 从build_train_data.py生成的图片中读取数据，用于训练
def get_screen_shot_file_data(filepath):
    # img_data = tf.image.decode_jpeg(tf.gfile.FastGFile(filepath, 'rb').read())
    # img_data_gray = tf.image.rgb_to_grayscale(img_data)
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img, dtype='float32')
    x_in = np.reshape(img, [100, 100, 1])

    for i in range(len(x_in)):
        for j in range(len(x_in[i])):
            x_in[i][j][0] /= 255

    return [x_in], [[int(filepath.split('_')[-1].split('.')[0]) / 1000]]


def sortByTime(dirpath):
    a = [s for s in os.listdir(dirpath)
        if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: s.split('_')[0])
    return a

# 开始训练
def start_train(sess):
    path = './records/';
    dirs = os.listdir(path);
    print(dirs)
    print('总记录数：', len(dirs))
    batch = 0
    total_batch = 30
    while batch < total_batch:
        dir_index = 0
        for record in dirs:
            print(record)
            dir_index += 1
            touch_time_arr = []
            # 忽略掉数据少的
            if len(os.listdir(path + record)) < 8:
                print('忽略：', record)
                continue
            images = sortByTime(path + record)
            print(images)
            for img in images[:-6]:
                if img.endswith('.jpg') and img.find('_') > 0:
                    filepath = path + record + '/' + img
                    x_in, y_out = get_screen_shot_file_data(filepath)
                    # print(x_in, y_out)
                    # break
                    # ————————————————这里只是打印出来看效果——————————————————
                    # y_result 神经网络自己算出来的按压时间
                    y_result = sess.run(y_fc3, feed_dict={x: x_in, keep_prob: 1})
                    # loss 计算损失
                    loss = sess.run(cross_entropy, feed_dict={y_fc3: y_result, y_: y_out})
                    touch_time_arr.append(loss)
                    ctime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(ctime, '\t', str(batch) + '/' + str(total_batch), '\t', str(dir_index) + '/' + str(len(dirs)), '\t', filepath)
                    print('origin:', y_out[0][0])
                    print('result:', y_result[0][0])
                    print("loss:", '{0:.10f}'.format(loss))
                    # —————————————————————————————————————————————————————
                    # 使用x_in，y_out训练
                    sess.run(train_step, feed_dict={x: x_in, y_: y_out, keep_prob: 0.6, learn_rate: 0.00002})
            
            saveLoss('./time.npz', touch_time_arr)            
            saver_init.save(sess, "./model/mode.mod")
        batch += 1
    print('训练结束！')

#训练单局游戏的图
def train_one(sess, folder, batch):
    touch_time_arr = []
    # 忽略掉数据少的
    if len(os.listdir(folder)) < 8:
        print('忽略：', folder)
        return
    images = sortByTime(folder)
    print('总图片数：', len(images))
    print(images)
    for index in range(batch):
        img_index = 1;
        for img in images[:-6]:
            if img.endswith('.jpg') and img.find('_') > 0:
                filepath = folder + '/' + img
                x_in, y_out = get_screen_shot_file_data(filepath)
                # print(x_in, y_out)
                # break
                # ————————————————这里只是打印出来看效果——————————————————
                # y_result 神经网络自己算出来的按压时间
                y_result = sess.run(y_fc3, feed_dict={x: x_in, keep_prob: 1})
                # loss 计算损失
                loss = sess.run(cross_entropy, feed_dict={y_fc3: y_result, y_: y_out})
                touch_time_arr.append(loss)
                ctime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(ctime, '\t', str(index) + '/' + str(batch), '\t', str(img_index) + '/' + str(len(images)), "\t", filepath)
                print('origin:', y_out[0][0])
                print('result:', y_result[0][0])
                print("loss:", '{0:.10f}'.format(loss))
                # —————————————————————————————————————————————————————
                # 使用x_in，y_out训练
                sess.run(train_step, feed_dict={x: x_in, y_: y_out, keep_prob: 0.6, learn_rate: 0.00002})
            img_index += 1
    saveLoss('./time.npz', touch_time_arr)            
    saver_init.save(sess, "./model/mode.mod")

# 开始玩耍
def start_play(sess):
    folder =  './records/' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.mkdir(folder);
    while True:
        print("----------------------------")
        x_in = get_screen_shot()
        ctime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        shutil.copyfile('./jump_temp.jpg', folder + '/' + ctime + '.jpg');
        shutil.copyfile('./jump_temp.png', folder + '/' + ctime + '.png');
        if has_die(x_in):
            print("died!")
            train_one(sess, folder, 5)
            print('训练完成，创建新目录开始下一局：')
            # folder = './records/' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            restart()
            return

        # 神经网络的输出
        y_result = sess.run(y_fc3, feed_dict={x: x_in, keep_prob: 1})
        if y_result[0][0] < 0:
            y_result[0][0] = 0
        
        touch_time = int(y_result[0][0] * 1000)

        # rdn_t = random.randrange(20, 30);
        os.rename(folder + '/' + ctime + '.jpg', folder + '/' + ctime + '_' + str(touch_time) + '.jpg')
        os.rename(folder + '/' + ctime + '.png', folder + '/' + ctime + '_' + str(touch_time) + '.png')
        
        print("touch time: ", touch_time, "ms")
        jump(touch_time)
        time.sleep(touch_time / 1000 + random.randrange(40, 80) / 100)

def saveLoss(filepath, data):
    if os.path.exists(filepath) == False:
        # npr = np.array(data)
        np.savez(filepath, array=data)
    else:
        result = np.load(filepath)['array'].tolist()
        result = result + data
        np.savez(filepath, array=result)


# 区分是train还是play
IS_TRAINING = False
# with tf.device('/gpu:0'):
with tf.Session() as sess:
    sess.run(tf_init)
    model_path = './model/'
    if len(os.listdir(model_path)) > 0:
        saver_init.restore(sess, model_path + 'mode.mod')
    if IS_TRAINING:
        # while True:
        # saveLoss('./time.npz', [1, 2, 3])
        start_train(sess)
    else:
        # test()
        # while True:
        start_play(sess)
            # pass



