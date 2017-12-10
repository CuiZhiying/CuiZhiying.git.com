---
title: CNN in tensorflow
date: 2017-12-03 21:30:41
categories: Udacity
tags: CNN
---

#### 引言
Udacity上的[深度学习课程](https://classroom.udacity.com/courses/ud730/lessons/6377263405/concepts/66010388990923)学习到了第四大节,卷积神经网络.这一部分使用到的数据与前面几节是一样的,都是notMINST的数据,而且就是第一章数据预处理后得到的相对干净的可以直接拿来用的数据.

这一章当中,老师讲到我们做图像处理的时候选择使用绝大多数时候都是使用卷积神经网络,而不是普通的全连接的神经网络,因为我们已经明确我们的数据类型就是数字图像,所以我们要充分使用到这些图像的特征.我想这种充分利用数据特征的思想,在我们的学习和探索的过程当中非常具有指导意义.

闲话不说,放码过来吧!

#### 数据集的提取
首先添加头文件,事实上这一块的代码应该是在后面需要用到的过程中再作添加的,但是为了保证该blog的代码可以重新复制粘贴之后可以使用,所以我就全部添加进来了.阅读时可先跳过这一段
``` python
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
```
然后就是使用Python中的`pickle()`函数恢复数据集对象,文件路径自行替换.
``` python
pickle_file = "../udacity-class3/notMNIST.pickle"

with open( pickle_file,"rb" ) as f:
    save = pickle.load(f)
    train_dataset = save["train_dataset"]
    train_labels  = save["train_labels"]
    valid_dataset = save["valid_dataset"]
    valid_labels  = save["valid_labels"]
    test_dataset  = save["test_dataset"]
    test_labels   = save["test_labels"]
    del save
    print("Training set   :", train_dataset.shape, train_labels.shape)
    print("Validation set :", valid_dataset.shape, valid_labels.shape)
    print("Test set       :", test_dataset.shape,  test_labels.shape)
```
到此,代码的运行结果如下(可以看到和计算数据集维度):
```
Training set   : (200000, 28, 28) (200000,)
Validation set : (10000, 28, 28) (10000,)
Test set       : (10000, 28, 28) (10000,)
```
上面得到的仅仅是我们的原始数据集,还不是十分符合tensorflow中`tf.nn.conv2d`中数据格式(符合要求的数据集形式应该是 [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数],详情见下文解释),所以对数据进行进一步的格式化处理.
``` python
image_size   = 28
num_labels   = 10
num_channels = 1  # grayscale

import numpy as np

def reformat( dataset, labels):
    # 将数据增加了一个维度,也就是增加了第四维channel,值为1
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    # 将便签扩展,便签所在位为1,其余为0,参见第三节内容
    labels  = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat( train_dataset, train_labels )
valid_dataset, valid_labels = reformat( valid_dataset, valid_labels )
test_dataset,  test_labels  = reformat( test_dataset,  test_labels  )

print(" Training set   : ", train_dataset.shape, train_labels.shape )
print(" Validation set : ", valid_dataset.shape, valid_labels.shape )
print(" Test set       : ", test_dataset.shape,  test_labels.shape  )
```
得到的结果如下:
``` python
 Training set   :  (200000, 28, 28, 1) (200000, 10)
 Validation set :  (10000, 28, 28, 1) (10000, 10)
 Test set       :  (10000, 28, 28, 1) (10000, 10)
```
注意到,我们的数据是黑白图像,所以channel是1.自此,数据整理完毕
#### 构建一个简单的卷积神经网络
我们将要构建的卷积神经网络超级无敌简单,就只有两个卷积层和一个全连接层.

在这里,我们先定义一个计算精确度的函数`accuracy()`,思想很简单就是准确预测的结果个数和所有预测次数的比值:
``` python
def accuracy( predictions, labels ):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
           / predictions.shape[0])
```
接着正式开始构建卷积神经网络
``` python
batch_size = 16           # SGD,每次投喂的数据个数
patch_size = 5            # 卷积核的维度大小
depth      = 16           # 卷积层的卷积核个数
num_hidden = 64           # 全连接层的神经元个数

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels)
    )
    tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset  = tf.constant(test_dataset)

    # Variables
    # 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有
    # [filter_height, filter_width,  in_channels, out_channels]这样的shape，具体含义是
    # [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，
    # 有一个地方需要注意，第三维in_channels，就是参数input的第四维
    # tf.nn.conv2d()返回值,哇,高维的数据好难理解和计算呀
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases  = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases  = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev = 0.1))
    layer3_biases  = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    layer4_biases  = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Models.
    def model(data):
        conv   = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding="SAME", use_cudnn_on_gpu=False)
        hidden = tf.nn.relu(conv + layer1_biases)
        conv   = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding="SAME", use_cudnn_on_gpu=False)
        hidden = tf.nn.relu(conv + layer2_biases)
        shape  = hidden.get_shape().as_list()
        reshape = tf.reshape( hidden, [shape[0], shape[1]*shape[2]*shape[3]])
        hidden = tf.nn.relu(tf.matmul( reshape, layer3_weights) + layer3_biases )
        return tf.matmul( hidden, layer4_weights ) + layer4_biases

    logits = model(tf_train_dataset)
    loss   = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predicitons for the training, validation, and test data.
    train_prediciton = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction  = tf.nn.softmax(model(tf_test_dataset ))
```
在构建这一个如此简答的卷积神经网络花了我好多时间,其中很大一个原因就是卷积层的计算和理解.一个四维的数据( ⊙o⊙ )我花了不少时间来研究了一下构建卷积神经网络的核心函数`tf.nn.conv2d()`,理解其输入和输出.其详细的探索过程,参见本人的另外一篇拙文[探索tf.nn.conv2d()](/tf-nn-conv2d.md)
#### 训练神经网络
训练的过程就是一个普通的随机梯度下降法的实现过程,很容易理解.
``` python
num_steps = 1001

with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data   = train_dataset[offset: (offset + batch_size), :, :, :]
        batch_labels = train_labels[offset: (offset + batch_size), :]
        feed_dict    = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
        [optimizer, loss, train_prediciton], feed_dict = feed_dict)
        if(step % 50 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy : %.1f%%"   % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%"   % accuracy(
                valid_prediction.eval(), valid_labels))
            print("Test accuracy      : %.1f%%"   % accuracy(
                test_prediction.eval(),  test_labels ))
    print("Test accuracy      : %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```
该训练过程产出的训练结果如下所示:
```
Initialized
Minibatch loss at step 0: 3.185863
Minibatch accuracy : 12.5%
Validation accuracy: 10.0%
Test accuracy      : 10.0%
Minibatch loss at step 50: 2.149336
Minibatch accuracy : 18.8%
Validation accuracy: 25.5%
Test accuracy      : 26.4%
Minibatch loss at step 100: 1.362517
Minibatch accuracy : 50.0%
Validation accuracy: 67.0%
Test accuracy      : 74.6%
Minibatch loss at step 150: 0.608616
Minibatch accuracy : 75.0%
Validation accuracy: 74.3%
Test accuracy      : 81.9%
Minibatch loss at step 200: 1.148737
Minibatch accuracy : 75.0%
Validation accuracy: 73.3%
Test accuracy      : 81.2%
Minibatch loss at step 250: 0.786106
Minibatch accuracy : 62.5%
Validation accuracy: 76.2%
Test accuracy      : 83.6%
Minibatch loss at step 300: 0.670637
Minibatch accuracy : 68.8%
Validation accuracy: 77.5%
Test accuracy      : 85.2%
Minibatch loss at step 350: 1.414526
Minibatch accuracy : 62.5%
Validation accuracy: 77.3%
Test accuracy      : 84.5%
Minibatch loss at step 400: 0.885400
Minibatch accuracy : 75.0%
Validation accuracy: 79.0%
Test accuracy      : 86.2%
Minibatch loss at step 450: 0.200270
Minibatch accuracy : 93.8%
Validation accuracy: 79.5%
Test accuracy      : 86.5%
Minibatch loss at step 500: 1.044583
Minibatch accuracy : 75.0%
Validation accuracy: 78.2%
Test accuracy      : 85.3%
Minibatch loss at step 550: 0.497158
Minibatch accuracy : 87.5%
Validation accuracy: 80.8%
Test accuracy      : 88.3%
Minibatch loss at step 600: 0.612175
Minibatch accuracy : 87.5%
Validation accuracy: 79.9%
Test accuracy      : 87.0%
Minibatch loss at step 650: 0.874741
Minibatch accuracy : 81.2%
Validation accuracy: 81.0%
Test accuracy      : 88.4%
Minibatch loss at step 700: 0.810160
Minibatch accuracy : 75.0%
Validation accuracy: 80.8%
Test accuracy      : 88.2%
Minibatch loss at step 750: 0.436342
Minibatch accuracy : 81.2%
Validation accuracy: 81.7%
Test accuracy      : 89.0%
Minibatch loss at step 800: 0.025083
Minibatch accuracy : 100.0%
Validation accuracy: 81.9%
Test accuracy      : 89.1%
Minibatch loss at step 850: 0.135296
Minibatch accuracy : 100.0%
Validation accuracy: 81.6%
Test accuracy      : 89.0%
Minibatch loss at step 900: 1.040877
Minibatch accuracy : 68.8%
Validation accuracy: 81.8%
Test accuracy      : 89.2%
Minibatch loss at step 950: 0.581194
Minibatch accuracy : 75.0%
Validation accuracy: 81.9%
Test accuracy      : 89.3%
Minibatch loss at step 1000: 0.379073
Minibatch accuracy : 93.8%
Validation accuracy: 81.9%
Test accuracy      : 89.6%
Test accuracy      : 89.6%
```
该训练结果,嗯,说实话,并不是很好,还不如之前训练的普通的三层神经网络.毕竟这个网络结果太简单了,还有很多的改进空间.
#### 添加池化层
上面的卷积神经网络是使用卷积核的步长来压缩数据空间的,而我们普遍的卷积神经网络都是使用池化层来进行该步骤哦.在tensorflow中池化层所使用的函数有一个常用的是[`tf.nn.max_pool()`](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool),在这里只需要修改一下卷积神经网络构建部分的代码即可
``` python
batch_size = 16
patch_size = 5
depth      = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels)
    )
    tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset  = tf.constant(test_dataset)
    
    # Variables
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases  = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases  = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev = 0.1))
    layer3_biases  = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    layer4_biases  = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    
    # Models.
    def model(data):
        # 卷积中strides修改为1
        conv   = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=False)
        hidden = tf.nn.relu(conv + layer1_biases)

        # 添加了池化层,池化层的核大小为2*2,步长为2
        pooling = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
        # 同上
        conv   = tf.nn.conv2d(pooling, layer2_weights, [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=False)
        hidden = tf.nn.relu(conv + layer2_biases)
        # 同上
        pooling = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
        shape  = pooling.get_shape().as_list()
        reshape = tf.reshape( hidden, [shape[0], shape[1]*shape[2]*shape[3]])
        hidden = tf.nn.relu(tf.matmul( reshape, layer3_weights) + layer3_biases )
        return tf.matmul( hidden, layer4_weights ) + layer4_biases
    
    logits = model(tf_train_dataset)
    loss   = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    
    # Predicitons for the training, validation, and test data.
    train_prediciton = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction  = tf.nn.softmax(model(tf_test_dataset ))
```
这只是一个简单修改,具体修改内容可以看注释进行对比,然后训练添加了池化层的模型.该训练的过程跟上面的卷积神经网络一毛一样,仅用来对比训练结果.
``` python
num_steps = 1001

with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data   = train_dataset[offset: (offset + batch_size), :, :, :]
        batch_labels = train_labels[offset: (offset + batch_size), :]
        feed_dict    = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
        [optimizer, loss, train_prediciton], feed_dict = feed_dict)
        if(step % 50 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy : %.1f%%"   % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%"   % accuracy(
                valid_prediction.eval(), valid_labels))
            print("Test accuracy      : %.1f%%"   % accuracy(
                test_prediction.eval(),  test_labels ))
    print("Test accuracy      : %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```
得到的训练结果如下所示:
```
Initialized
Minibatch loss at step 0: 3.369772
Minibatch accuracy : 6.2%
Validation accuracy: 10.0%
Test accuracy      : 10.0%
Minibatch loss at step 50: 1.835429
Minibatch accuracy : 25.0%
Validation accuracy: 49.5%
Test accuracy      : 53.8%
Minibatch loss at step 100: 1.258682
Minibatch accuracy : 68.8%
Validation accuracy: 69.1%
Test accuracy      : 76.2%
Minibatch loss at step 150: 0.825770
Minibatch accuracy : 81.2%
Validation accuracy: 75.6%
Test accuracy      : 83.1%
Minibatch loss at step 200: 1.012264
Minibatch accuracy : 75.0%
Validation accuracy: 74.3%
Test accuracy      : 81.6%
Minibatch loss at step 250: 0.624423
Minibatch accuracy : 81.2%
Validation accuracy: 76.8%
Test accuracy      : 84.0%
Minibatch loss at step 300: 0.578071
Minibatch accuracy : 87.5%
Validation accuracy: 78.5%
Test accuracy      : 85.8%
Minibatch loss at step 350: 1.048332
Minibatch accuracy : 68.8%
Validation accuracy: 78.8%
Test accuracy      : 85.8%
Minibatch loss at step 400: 0.842998
Minibatch accuracy : 81.2%
Validation accuracy: 79.3%
Test accuracy      : 86.6%
Minibatch loss at step 450: 0.198143
Minibatch accuracy : 93.8%
Validation accuracy: 80.5%
Test accuracy      : 87.5%
Minibatch loss at step 500: 1.044419
Minibatch accuracy : 75.0%
Validation accuracy: 80.7%
Test accuracy      : 88.3%
Minibatch loss at step 550: 0.667056
Minibatch accuracy : 87.5%
Validation accuracy: 80.8%
Test accuracy      : 88.1%
Minibatch loss at step 600: 0.454019
Minibatch accuracy : 87.5%
Validation accuracy: 81.6%
Test accuracy      : 88.8%
Minibatch loss at step 650: 0.988948
Minibatch accuracy : 68.8%
Validation accuracy: 82.0%
Test accuracy      : 89.5%
Minibatch loss at step 700: 0.723278
Minibatch accuracy : 81.2%
Validation accuracy: 81.5%
Test accuracy      : 88.7%
Minibatch loss at step 750: 0.365816
Minibatch accuracy : 87.5%
Validation accuracy: 82.5%
Test accuracy      : 89.9%
Minibatch loss at step 800: 0.066661
Minibatch accuracy : 100.0%
Validation accuracy: 82.8%
Test accuracy      : 89.6%
Minibatch loss at step 850: 0.204858
Minibatch accuracy : 93.8%
Validation accuracy: 82.5%
Test accuracy      : 90.1%
Minibatch loss at step 900: 1.044320
Minibatch accuracy : 68.8%
Validation accuracy: 82.4%
Test accuracy      : 89.7%
Minibatch loss at step 950: 0.612166
Minibatch accuracy : 81.2%
Validation accuracy: 83.5%
Test accuracy      : 90.7%
Minibatch loss at step 1000: 0.502493
Minibatch accuracy : 87.5%
Validation accuracy: 82.9%
Test accuracy      : 90.4%
Test accuracy      : 90.4%
```
结果有了一点点的改善,改善不是特别的明显.
#### 实现一个经典的LeNet5网络结构
LeNet5是一个十分经典的卷积神经网络,其机构如下所示:
![LeNet5.png](LeNet5.png)
结果如下
![asamples.gif](asamples.gif)
根据该图,我们只需要修改tensorflow中graph的代码,即可构造一个类似于的LeNet5网络架构的网络了.
注意,修改了网络结构需要重新计算最后生成图片的大小.
``` python
batch_size = 16
patch_size = 5
depth      = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels)
    )
    tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset  = tf.constant(test_dataset)
    
    # Variables
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases  = tf.Variable(tf.zeros([depth]))
    # add the max_pool arguments
    # l1_pool_weight = tf.Variable()
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases  = tf.Variable(tf.constant(1.0, shape=[depth]))
    # 重新计算生成图片的大小
    def get_conv_pool_size( image_size, patch_size):
        return (image_size - patch_size + 1) // 2
    layer3_weights = tf.Variable(tf.truncated_normal(
        [get_conv_pool_size(get_conv_pool_size(image_size, patch_size), patch_size) ** 2 * depth, \
         num_hidden],\
        stddev = 0.1))
    layer3_biases  = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    layer4_biases  = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    
    # Models.
    def model(data):
        # input data is 28 * 28
        conv1    = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding="VALID", use_cudnn_on_gpu=False)
        hidden1  = tf.nn.relu(conv1 + layer1_biases)
        # hidden1    is 24 * 24
        pooling1 = tf.nn.max_pool(hidden1, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
        # pooling1   is 12 * 12
        conv2    = tf.nn.conv2d(pooling1, layer2_weights, [1, 1, 1, 1], padding="VALID", use_cudnn_on_gpu=False)
        hidden2  = tf.nn.relu(conv2 + layer2_biases)
        # hidden2    is 8 * 8
        pooling2 = tf.nn.max_pool(hidden2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
        # pooling2   is 4 * 4
        shape  = pooling2.get_shape().as_list()
        reshape = tf.reshape( pooling2, [shape[0], shape[1]*shape[2]*shape[3]])
        hidden = tf.nn.relu(tf.matmul( reshape, layer3_weights) + layer3_biases )
        return tf.matmul( hidden, layer4_weights ) + layer4_biases
    
    logits = model(tf_train_dataset)
    loss   = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    
    # Predicitons for the training, validation, and test data.
    train_prediciton = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction  = tf.nn.softmax(model(tf_test_dataset ))
```
训练该生成模型,因为该模型收敛得慢一些,所以,我增加了训练的次数,从1001次改成了10001次.并且可以观察到,该模型已经逐渐趋于稳定了.
训练结果如下所示:
```
Initialized
Minibatch loss at step 0: 3.063954
Minibatch accuracy : 6.2%
Validation accuracy: 10.0%
Test accuracy      : 10.0%
Minibatch loss at step 500: 0.920886
Minibatch accuracy : 75.0%
Validation accuracy: 78.8%
Test accuracy      : 85.7%
Minibatch loss at step 1000: 0.605112
Minibatch accuracy : 87.5%
Validation accuracy: 81.8%
Test accuracy      : 89.3%
Minibatch loss at step 1500: 0.555171
Minibatch accuracy : 81.2%
Validation accuracy: 84.8%
Test accuracy      : 91.0%
Minibatch loss at step 2000: 0.826236
Minibatch accuracy : 75.0%
Validation accuracy: 84.6%
Test accuracy      : 91.4%
Minibatch loss at step 2500: 0.663057
Minibatch accuracy : 81.2%
Validation accuracy: 85.4%
Test accuracy      : 91.7%
Minibatch loss at step 3000: 0.747276
Minibatch accuracy : 81.2%
Validation accuracy: 85.5%
Test accuracy      : 91.9%
Minibatch loss at step 3500: 0.708025
Minibatch accuracy : 75.0%
Validation accuracy: 86.2%
Test accuracy      : 92.7%
Minibatch loss at step 4000: 0.046353
Minibatch accuracy : 100.0%
Validation accuracy: 86.5%
Test accuracy      : 93.1%
Minibatch loss at step 4500: 0.519293
Minibatch accuracy : 81.2%
Validation accuracy: 87.0%
Test accuracy      : 93.2%
Minibatch loss at step 5000: 0.568638
Minibatch accuracy : 81.2%
Validation accuracy: 86.8%
Test accuracy      : 93.0%
Minibatch loss at step 5500: 0.231502
Minibatch accuracy : 93.8%
Validation accuracy: 87.2%
Test accuracy      : 93.4%
Minibatch loss at step 6000: 0.876349
Minibatch accuracy : 75.0%
Validation accuracy: 87.0%
Test accuracy      : 93.4%
Minibatch loss at step 6500: 0.617021
Minibatch accuracy : 81.2%
Validation accuracy: 87.5%
Test accuracy      : 93.5%
Minibatch loss at step 7000: 0.159559
Minibatch accuracy : 100.0%
Validation accuracy: 87.6%
Test accuracy      : 93.8%
Minibatch loss at step 7500: 0.437542
Minibatch accuracy : 87.5%
Validation accuracy: 87.1%
Test accuracy      : 93.3%
Minibatch loss at step 8000: 0.537314
Minibatch accuracy : 75.0%
Validation accuracy: 88.1%
Test accuracy      : 93.9%
Minibatch loss at step 8500: 0.458627
Minibatch accuracy : 87.5%
Validation accuracy: 88.2%
Test accuracy      : 94.3%
Minibatch loss at step 9000: 0.507361
Minibatch accuracy : 75.0%
Validation accuracy: 88.0%
Test accuracy      : 93.8%
Minibatch loss at step 9500: 0.630668
Minibatch accuracy : 68.8%
Validation accuracy: 87.8%
Test accuracy      : 93.5%
Minibatch loss at step 10000: 0.416923
Minibatch accuracy : 87.5%
Validation accuracy: 88.4%
Test accuracy      : 94.1%
Test accuracy      : 94.1%
```
从该结果可以明显看到图像识别的效率得到了显著的提高.
#### 总结
通过仔细分析和动手构造一些简单的经典的神经网络,可以让我们更加直观深入得去了解熟悉和应用我们的知识,发现更多的细节.
主要耗时的地方在于四维的数据,我不知道怎么样进行运算,所以花了很多时间去探索`tf.nn.conv2d()`这一个函数
