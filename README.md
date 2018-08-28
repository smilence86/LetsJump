修复模型训练bug，在训练迭代过程中调用变量tf.image会导致内存暴涨，训练越来越慢。

引入batch小批量梯度下降，当batch_size=1时为随机梯度下降。

引入batch_norm，正则化层内数据分布，加快梯度下降。

训练迭代过程中，避免多次调用sess.run()，防止变量运行多次导致数据混乱，如果要运行多个变量则调用一次sess.run()，传入多个参数得到批量结果。

600张样本图片，当batch_size=10，大约训练500 EPOCH时loss趋势图：



# LetsJump
Android手机上自动玩微信跳一跳的TensorFlow实现
![](https://github.com/zhanyongsheng/raw/blob/master/LetsJump/pic.jpg)  
<br>
### build_train_data.py
用于手动生成训练数据，生成的数据保存在train_data文件夹下，为图片和对应的按压时间。
<br>
### lets_jump.py
用于Training/Play，目前save文件夹下已保存了训练两个多小时的模型，可以直接拿来用。<br>
把手机连上电脑，确保电脑上已配置adb环境，打开跳一跳游戏界面，运行文件，开始愉快的玩耍。<br>
<br>
以上训练数据和代码都是基于1920*1080分辨率手机
<br><br>
博客地址：[用TensorFlow做一个玩微信跳一跳的AI](http://blog.csdn.net/zhanys_7/article/details/78940763)

