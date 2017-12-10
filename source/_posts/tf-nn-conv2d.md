---
title: 探索tf.nn.conv2d()
date: 2017-12-04 02:13:43
categories: tensorflow
tags: CNN
---
#### 前言
我在用tensorflow来实现卷积神经网络的时候,遇到的最大的问题就是对tf.nn.conv2d()这个函数的使用.整个使用过程我都觉得是稀里糊涂的.
#### 黑盒子探索
我在blog上找到了一个不错的指导实践教程[戳这里](http://www.cnblogs.com/welhzh/p/6607581.html),上面的讲解很清晰(详细内容请阅读原作者博客,我就不复制粘贴了).
讲解明则明已,发现上面代码重复率超级高,所以自己又稍微整理了一下
``` python
import tensorflow as tf

def ones_conv2d( image_size  = [1, 1, 1, 1], \
                 filter_size = [1, 1, 1, 1], \
                 strides     = [1, 1, 1, 1], \
                 padding     = "VALID",      \
                 *,
                 show_result = True):
    image  = tf.Variable( tf.ones( image_size ) )
    filter = tf.Variable( tf.ones( filter_size) )
    conv2d = tf.nn.conv2d(image, filter, strides = strides, padding = padding)
    with tf.Session() as session:
        session.run(tf.variables_initializer([image, filter]))
        feature = session.run(conv2d)

    if( show_result ):
        print("\ninput image size : ", image_size )
        print("filter size      : ", filter_size )
        print("strides          : ", strides )
        print("output feature is: ")
        print(feature)
    return feature

# ones_conv2d([1, 3, 3, 5], [1, 1, 5, 1], [1, 1, 1, 1], show_result = False)
# ones_conv2d([1, 3, 3, 5], [1, 1, 5, 1], [1, 1, 1, 1] )
# ones_conv2d([1, 5, 5, 5], [3, 3, 5, 1], [1, 1, 1, 1], padding = "SAME" )
# ones_conv2d([1, 5, 5, 5], [3, 3, 5, 1], [1, 2, 2, 1], padding = "SAME" )
ones_conv2d([3, 5, 5, 5], [3, 3, 5, 7], [1, 2, 2, 1], padding = "SAME" )

```
程序执行结果如下:
```
input image size :  [3, 5, 5, 5]
filter size      :  [3, 3, 5, 7]
strides          :  [1, 2, 2, 1]
output feature is:
[[[[ 20.  20.  20.  20.  20.  20.  20.]
   [ 30.  30.  30.  30.  30.  30.  30.]
   [ 20.  20.  20.  20.  20.  20.  20.]]

  [[ 30.  30.  30.  30.  30.  30.  30.]
   [ 45.  45.  45.  45.  45.  45.  45.]
   [ 30.  30.  30.  30.  30.  30.  30.]]

  [[ 20.  20.  20.  20.  20.  20.  20.]
   [ 30.  30.  30.  30.  30.  30.  30.]
   [ 20.  20.  20.  20.  20.  20.  20.]]]


 [[[ 20.  20.  20.  20.  20.  20.  20.]
   [ 30.  30.  30.  30.  30.  30.  30.]
   [ 20.  20.  20.  20.  20.  20.  20.]]

  [[ 30.  30.  30.  30.  30.  30.  30.]
   [ 45.  45.  45.  45.  45.  45.  45.]
   [ 30.  30.  30.  30.  30.  30.  30.]]

  [[ 20.  20.  20.  20.  20.  20.  20.]
   [ 30.  30.  30.  30.  30.  30.  30.]
   [ 20.  20.  20.  20.  20.  20.  20.]]]


 [[[ 20.  20.  20.  20.  20.  20.  20.]
   [ 30.  30.  30.  30.  30.  30.  30.]
   [ 20.  20.  20.  20.  20.  20.  20.]]

  [[ 30.  30.  30.  30.  30.  30.  30.]
   [ 45.  45.  45.  45.  45.  45.  45.]
   [ 30.  30.  30.  30.  30.  30.  30.]]

  [[ 20.  20.  20.  20.  20.  20.  20.]
   [ 30.  30.  30.  30.  30.  30.  30.]
   [ 20.  20.  20.  20.  20.  20.  20.]]]]
```
上面共分为3大组,每一组有3个矩阵,每个矩阵有3行,每一行有7个数字.
理解的方式就是3大组表示输入了3张图片,分别进行卷积运算.每一大组代表了每一张图片的卷积运算结果.
好,我们来看其中一组的卷积运算结果:一组有三个矩阵,表示这一张图片的特征图像是有3列,这三个矩阵呢每一个都有三行,代表了这一张图片的特征图像有三行.矩阵的每一行都有7列,代表了他们将会组成7个特征图像.
所以上面的运算得到的一张图片的7个特征图像都是这样子的.
```
[ 20. 30. 20.]
[ 30. 45. 30.] 
[ 20. 30. 20.]
```
#### TO do: 读源码
好吧,这是一个程序员的笑话.待我研究好了这个部分的源代码,再来总结分享.
