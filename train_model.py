# -*- coding: utf-8 -*-
import math, random, time as time, datetime, shutil, os, cv2
import tensorflow as tf
import numpy as np


def weight_variable(shape, std):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=std)
    return tf.Variable(initial)

def bias_variable(shape, std):
    initial = tf.constant(std, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, b, s=1):
    x = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    out_size = W.get_shape().as_list()[3]
    # print(out_size)
    x = batch_norm(x, out_size)
    return tf.nn.leaky_relu(x)

def maxPool(x, k=2):
    return tf.nn.max_pool2d(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def avgPool(x, k=2):
    return tf.nn.avg_pool2d(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

EPOCH = 30
width=100               #输入图片宽度
height=100              #输入图片高度
channels=1              #输入图片通道数（RGB）
batch_size = 5
Learn_rate = 0.00002
# 输入：100*100的灰度图片，前面的None是batch size
x = tf.placeholder(tf.float32, shape=[None, width, height, channels])
# 输出：一个浮点数，就是按压时间，单位ms
y = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)
learn_rate = tf.placeholder(tf.float32)

weights = {
    #
    'wc1': weight_variable([3, 3, 1, 32], 0.1),
    #
    'wc2': weight_variable([3, 3, 32, 64], 0.1),
    #
    'wc3': weight_variable([3, 3, 64, 128], 0.1),
    #
    'wc4': weight_variable([3, 3, 128, 256], 0.1),
    #
    'wc5': weight_variable([3, 3, 256, 512], 0.1),
    #1×1卷积核，降低输出维度
    'wc6': weight_variable([1, 1, 512, 256], 0.1),
    # fully connected, 
    'w_fc1': weight_variable([2*2*256, 1024], 0.1),
    # fully connected, 
    'w_fc2': weight_variable([1024, 512], 0.1),
    # 
    'out': weight_variable([512, 1], 0.1)
}

biases = {
    'bc1': bias_variable([32], 0.1),
    'bc2': bias_variable([64], 0.1),
    'bc3': bias_variable([128], 0.1),
    'bc4': bias_variable([256], 0.1),
    'bc5': bias_variable([512], 0.1),
    'bc6': bias_variable([256], 0.1),
    'b_fc1': bias_variable([1024], 0.1),
    'b_fc2': bias_variable([512], 0.1),
    'out': bias_variable([1], 0.1)
}

def batch_norm(x, n_out, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = mean_var_with_update()
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def conv_net(x, weights, biases, keep_prob):
    # x = tf.reshape(x, shape=[-1, 100, 100, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print(conv1.shape)
    max_pool1 = maxPool(conv1, k=2)
    print(max_pool1.shape)

    conv2 = conv2d(max_pool1, weights['wc2'], biases['bc2'])
    print(conv2.shape)
    max_pool2 = maxPool(conv2, k=2)
    print(max_pool2.shape)

    conv3 = conv2d(max_pool2, weights['wc3'], biases['bc3'])
    print(conv3.shape)
    max_pool3 = maxPool(conv3, k=2)
    print(max_pool3.shape)

    conv4 = conv2d(max_pool3, weights['wc4'], biases['bc4'])
    print(conv4.shape)
    max_pool4 = maxPool(conv4, k=2)
    print(max_pool4.shape)

    conv5 = conv2d(max_pool4, weights['wc5'], biases['bc5'])
    print(conv5.shape)
    max_pool5 = maxPool(conv5, k=2)
    print(max_pool5.shape)

    conv6 = conv2d(max_pool5, weights['wc6'], biases['bc6'])
    print(conv6.shape)
    max_pool6 = avgPool(conv6, k=2)
    print(max_pool6.shape)

    pool_flat = tf.reshape(max_pool6, [-1, weights['w_fc1'].get_shape().as_list()[0]])
    fc1 = tf.nn.leaky_relu(tf.matmul(pool_flat, weights['w_fc1']) + biases['b_fc1'])
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    fc2 = tf.nn.leaky_relu(tf.matmul(fc1_drop, weights['w_fc2']) + biases['b_fc2'])

    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

pred = conv_net(x, weights, biases, keep_prob)

# 因输出直接是时间值，回归问题，而不是分类概率，所以用平方差损失
diff = tf.subtract(y, pred)
square = tf.square(diff)
# sum_ = tf.reduce_sum(square)
cost = tf.reduce_mean(square)
# cost = tf.reduce_mean(tf.square(tf.subtract(pred, y)))
train_step = tf.compat.v1.train.AdamOptimizer(learn_rate).minimize(cost)

tf_init = tf.compat.v1.global_variables_initializer()
saver_init = tf.train.Saver()#dict(weights, **biases)


# 获取屏幕截图并转换为模型的输入
def get_screen_shot(folder):
    # 使用adb命令截图并获取图片
    os.system('adb shell screencap -p /sdcard/jump_temp.png')
    os.system('adb pull /sdcard/jump_temp.png .')
    img = cv2.imread('./jump_temp.png')
    w, h, chs = img.shape
    # 将图片压缩，并截取中间部分，截取后为100*100
    img = cv2.resize(img, (108, 192))
    # cv2.imshow('sss', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = img[50:150, 4:104]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    millisecond = int(round(time.time() * 1000))
    cv2.imwrite(folder + str(millisecond) + '.png', img)
    img = np.asarray(img, dtype='float32')
    x_in = np.reshape(img, [width, height, channels])

    # [0,255]转为[0,1]浮点
    for i in range(len(x_in)):
        for j in range(len(x_in[i])):
            x_in[i][j][0] /= 255

    # 因为输入shape有batch维度，所以还要套一层
    return [x_in]


# 按压press_time时间后松开，完成一次跳跃
def jump(press_time):

    xs = [random.uniform(840, 870) for _ in range(2)]
    ys = [random.uniform(1750, 1820) for _ in range(2)]
    
    cmd = 'adb shell input swipe {} {} {} {} {}'.format(
        int(xs[0]),
        int(ys[0]), 
        int(xs[1]), 
        int(ys[1]), 
        press_time
    )
    print(cmd)
    os.system(cmd)


# 判断是否游戏失败到分数页面
def has_die(x_in):
    # print(x_in[0])
    # 判断左上右上左下右下四个点的亮度
    # and (x_in[0][len(x_in[0]) - 1][0][0] < 0.4) and (x_in[0][len(x_in[0]) - 1][len(x_in[0][0]) - 1][0] < 0.4)
    if (x_in[0][0][10][0] < 0.4) and (x_in[0][0][len(x_in[0][0]) - 10][0] < 0.4):
        return True
    else:
        return False


# 游戏失败后重新开始，(540，1588)为1080*1920分辨率手机上重新开始按钮的位置
def restart():
    #adb shell wm size
    cmd = 'adb shell input swipe 550 1588 560 1598 {}'.format(int(random.uniform(35, 140)))
    os.system(cmd)
    time.sleep(3)


# 从build_train_data.py生成的图片中读取数据，用于训练
def get_screen_shot_file_data(filepath):
    # img_data = tf.image.decode_jpeg(tf.gfile.FastGFile(filepath, 'rb').read())
    # img_data_gray = tf.image.rgb_to_grayscale(img_data)
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img, dtype='float32')
    x_in = np.reshape(img, [width, height, channels])

    for i in range(len(x_in)):
        for j in range(len(x_in[i])):
            x_in[i][j][0] /= 255

    return x_in, [int(filepath.split('_')[-1].split('.')[0])]


def sortByTime(dirpath):
    a = [s for s in os.listdir(dirpath)
        if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: s.split('_')[0])
    return a

# 开始训练
def start_train(sess):
    path = './records/'
    dirs = os.listdir(path)
    dirs.sort()
    print(dirs)
    print('总局数：', len(dirs))
    batch = 0
    total_batch = 10
    while batch < total_batch:
        dir_index = 0
        dir_total = 5  #最后几局游戏
        for record in dirs[-dir_total:]:
            # train_one(sess, './records/2018-01-30 13:15:00', 30)
            print(record)
            dir_index += 1
            touch_time_arr = []
            # 忽略掉数据少的
            if len(os.listdir(path + record)) < 10:
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
                    y_result = sess.run(pred, feed_dict={x: x_in, keep_prob: 1})
                    # loss 计算损失
                    loss = sess.run(cross_entropy, feed_dict={pred: y_result, y_: y_out})
                    touch_time_arr.append(loss)
                    ctime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(ctime, '\t', str(batch) + '/' + str(total_batch), '\t', str(dir_index) + '/' + str((dir_total)), '\t', filepath)
                    print('origin:', y_out[0][0])
                    print('result:', y_result[0][0])
                    if loss > 0.2:
                        print("异常loss:", '{0:.10f}'.format(loss))
                    else:
                        print("loss:", '{0:.10f}'.format(loss))
                    # —————————————————————————————————————————————————————
                    # 使用x_in，y_out训练
                    sess.run(train_step, feed_dict={x: x_in, y_: y_out, keep_prob: 0.6, learn_rate: 0.00002})
            
            saveLoss('./time.npz', touch_time_arr)            
            saver_init.save(sess, "./model/mode.mod")
        batch += 1
    print('训练结束！')

#训练单局游戏的图
def train_one(sess, folder, epoch):
    # 忽略掉数据少的
    if len(os.listdir(folder)) < 30:
        print('忽略：', folder)
        return
    images = sortByTime(folder)[:]
    print('总样本数：', len(images))
    # print(images)
    total_page = math.ceil(len(images) / batch_size)
    print('总页数：', total_page)
    for e in range(epoch):
        loss_array = []
        for page in range(0, total_page):
            imgs = images[page * batch_size:page * batch_size + batch_size]
            batch_xs = []
            batch_ys = []
            for img in imgs[:]:
                if img.endswith('.jpg') and img.find('_') > 0:
                    filepath = folder + '/' + img
                    x_in, y_out = get_screen_shot_file_data(filepath)
                    # print(x_in, y_out)
                    batch_xs.append(x_in)
                    batch_ys.append(y_out)
            # ————————————————这里只是打印出来看效果——————————————————
            # 使用x_in，y_out训练
            d, s, loss, y_pred, NULL = sess.run([diff, square, cost, pred, train_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.6, learn_rate: Learn_rate})
            #输出进度
            ctime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(ctime, '\t', str(e) + '/' + str(epoch), '\t', str(page) + '/' + str(total_page), "\t", filepath)
            # print(d)
            # print(s)
            # y_pred 神经网络自己算出来的按压时间
            loss_array.append(loss)
            #对比结果
            print('origin:', [round(i[0], 0) for i in batch_ys])
            print('result:', [int(i[0]) for i in y_pred.tolist()])
            print("loss:", '{0:.10f}'.format(loss))
            # —————————————————————————————————————————————————————
        saveLoss('./loss.npz', loss_array)
        saver_init.save(sess, "./model/mode.mod")
    print('训练完成！')

# 开始玩耍
def start_play(sess):
    folder =  './records/' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.mkdir(folder)
    while True:
        print("----------------------------")
        x_in = get_screen_shot(folder)
        ctime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        shutil.copyfile('./jump_temp.jpg', folder + '/' + ctime + '.jpg');
        shutil.copyfile('./jump_temp.png', folder + '/' + ctime + '.png');
        if has_die(x_in):
            print('died!')
            # train_one(sess, folder, 5)
            restart()
            return

        # 神经网络的输出
        y_result = sess.run(pred, feed_dict={x: x_in, keep_prob: 1})
        if y_result[0][0] < 0:
            y_result[0][0] = 0
        
        touch_time = int(y_result[0][0])

        # rdn_t = random.randrange(20, 30);
        os.rename(folder + '/' + ctime + '.jpg', folder + '/' + ctime + '_' + str(touch_time) + '.jpg')
        os.rename(folder + '/' + ctime + '.png', folder + '/' + ctime + '_' + str(touch_time) + '.png')
        
        print("touch time: ", touch_time, "ms")
        jump(touch_time)
        time.sleep(touch_time / 1000 + random.randrange(800, 1500) / 1000)

def saveLoss(filepath, data):
    if os.path.exists(filepath) == False:
        np.savez(filepath, array=data)
    else:
        result = np.load(filepath)['array'].tolist()
        result = result + data
        np.savez(filepath, array=result)


# 区分是train还是play
IS_TRAINING = True
# IS_TRAINING = False
# with tf.device('/gpu:0'):
with tf.Session() as sess:
    sess.run(tf_init)
    model_path = './model/'
    if len(os.listdir(model_path)) > 0:
        saver_init.restore(sess, model_path + 'mode.mod')
    if IS_TRAINING:
        # while True:
        # x_in = get_screen_shot()
        # print(has_die(x_in))
        train_one(sess, './2018-01-30 13:15:00', EPOCH)
        # start_train(sess)
    else:
        while True:
            start_play(sess)
            # pass



