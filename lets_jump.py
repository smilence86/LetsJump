# -*- coding: utf-8 -*-

import os, random, datetime, shutil
import tensorflow as tf
import numpy as np
from PIL import Image
import time as time

# 区分是train还是play
IS_TRAINING = False


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

# drapout，play时为1训练时为0.6
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 学习率
learn_rate = tf.placeholder(tf.float32)

# 32 --> 1
W_fc2 = weight_variable([32, 1], 0.1)
b_fc2 = bias_variable([1], 0.1)
y_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 因输出直接是时间值，而不是分类概率，所以用平方损失
cross_entropy = tf.reduce_mean(tf.square(y_fc2 - y_))
train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

tf_init = tf.global_variables_initializer()

saver_init = tf.train.Saver({"W_conv1": W_conv1, "b_conv1": b_conv1,
                             "W_conv2": W_conv2, "b_conv2": b_conv2,
                             "W_fc1": W_fc1, "b_fc1": b_fc1,
                             "W_fc2": W_fc2, "b_fc2": b_fc2})


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
    cmd = 'adb shell input swipe 540 1588 540 1588 {}'.format(int(random.uniform(35, 98)))
    os.system(cmd)
    time.sleep(2)


# 从build_train_data.py生成的图片中读取数据，用于训练
def get_screen_shot_file_data(filecount):
    filename = "./train_data/" + str(filecount) + ".jpg"
    img_data = tf.image.decode_jpeg(tf.gfile.FastGFile(filename, 'rb').read())
    img_data_gray = tf.image.rgb_to_grayscale(img_data)
    x_in = np.asarray(img_data_gray.eval(), dtype='float32')

    for i in range(len(x_in)):
        for j in range(len(x_in[i])):
            x_in[i][j][0] /= 255

    return [x_in]


# 开始训练
def start_train(sess):
    # 读取应该按压的时间数组
    arr = np.load("./train_data/time.npz")["abc"].tolist()
    print("arr:", arr)
    count = len(arr)
    print("count:", count)

    # 图片文件名指示
    file_count = 0

    # 训练了多少次
    train_count = 0
    while True:

        # 所有图片都训练完了，从头开始
        if file_count >= count - 1:
            file_count = 0

        x_in = get_screen_shot_file_data(file_count)
        y_out = [[arr[file_count]]]

        # 每训练100个保存一次
        if train_count % 100 == 0:
            saver_init.save(sess, "./save/mode.mod")

        # ————————————————这里只是打印出来看效果——————————————————
        # y_result 神经网络自己算出来的按压时间
        y_result = sess.run(y_fc2, feed_dict={x: x_in, keep_prob: 1})
        # loss 计算损失
        loss = sess.run(cross_entropy, feed_dict={y_fc2: y_result, y_: y_out})
        print(str(train_count), "y_out:", y_out, "y_result:", y_result, "loss:", loss)
        # —————————————————————————————————————————————————————

        # 使用x_in，y_out训练
        sess.run(train_step, feed_dict={x: x_in, y_: y_out, keep_prob: 0.6, learn_rate: 0.00002})

        file_count = file_count + 1
        train_count = train_count + 1


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
            folder = './records/' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # os.mkdir(folder);
            restart()
            return

        # 神经网络的输出
        y_result = sess.run(y_fc2, feed_dict={x: x_in, keep_prob: 1})
        if y_result[0][0] < 0:
            y_result[0][0] = 0

        print("touch time: ", y_result[0][0] * 1000, "ms")

        touch_time = int(y_result[0][0] * 1000)
        os.rename(folder + '/' + ctime + '.jpg', folder + '/' + ctime + '_' + str(touch_time) + '.jpg')
        os.rename(folder + '/' + ctime + '.png', folder + '/' + ctime + '_' + str(touch_time) + '.png')
        jump(touch_time)
        time.sleep(touch_time / 1000 + 0.5)


def test():
    arr = np.load("./train_data/time.npz")["abc"].tolist()
    print(len(arr))
    print(arr)

with tf.Session() as sess:
    sess.run(tf_init)
    saver_init.restore(sess, "./save/mode.mod")
    if IS_TRAINING:
        while True:
            start_train(sess)
    else:
        # test()
        while True:
            start_play(sess)
            # pass


