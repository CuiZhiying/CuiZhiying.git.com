---
title: Udacity-note-book
date: 2017-10-13 19:13:31
categories:
tags: note-book
---

#### class1 从机器学习到深度学习
##### softmax函数

``` python
import numpy as np
def softmax( x ):
    return np.exp(x) / np.sum( np.exp(x), axis = 0 )
```

为什么axis = 0呢,没有搞清楚

如果将`x*10`的话, softmax的结果会更加趋向于1和0的极端分布,如果将`x/10`的话,结果会变得更加平均,没有区分度.注意这一点,我们在训练初期的时候,可以让x除以10,磨平他们之间的差距,在训练的后期,则可以将x乘以10,以增大其区分度

`1e-6`得到的是0.000006,记住这种高端大气的科学计数法

np.reshape(-1) 的意思是:numpy allow us to give one of new shape parameter as -1 (eg: (2,-1) or (-1,3) but not (-1, -1)). It simply means that it is an unknown dimension and we want numpy to figure it out. And numpy will figure this by looking at the 'length of the array and remaining dimensions' and making sure it satisfies the above mentioned criteria

https://stackoverflow.com/questions/36526035/understanding-applied-to-a-numpy-array
教我numpy的高级用法

tf.nn.softmax_coss_entropy_with_logits( labels = train_label, logits = logits)

等价于

y = tf.nn.softmax( logits )
coss_entropy = -tf.reduce_mean( tf_train_labels * tf.log( y ))


.run() 和 .eval()的区别:
https://stackoverflow.com/questions/33610685/in-tensorflow-what-is-the-difference-between-session-run-and-tensor-eval
基本等价,区别是.run()可以获取到多个向量的值,而.eval()只能获取单个数值.

紧身裤原理:紧身裤很合适,很美,却很难穿得上.因而人们会选择一些稍微宽松一点的裤子.
机器学习:给出一个非常合适的模型将会使得该模型非常难以训练.因而人们会提出一些宽松一点的模型,便于训练.要注意防止过拟合的问题.
正则化问题则是相对于弹力裤了,可以自己调节,很合适!!!

drop out : 随机丢弃一些之前激活层的一些数据,真的是非常的疯狂啊!!!但是呢,可以有效防止过拟合..为了保持丢弃后数据的平均值,所以要将剩下的数据进行放大处理.
