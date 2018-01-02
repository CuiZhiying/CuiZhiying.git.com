---
title: Lesson 1-- From machine learning to Deep learning
date: 2017-12-12 16:57:26
categories: [Udacity Deep Learning]
tags: [tensorflow, deep-learning, tutorial, note]
---

## 安装好环境
都是一些非常常用的包,自己电脑环境缺失的请用`pip install package-name`来装上,可以先不在这里看这些代码
``` python
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
```
## 从网上下载原始数据
使用代码进行下载的时候,有可能会被服务器拦截,最简单的方法就是双击点开[_**这里**_](http://yaroslavvb.com/upload/notMNIST/), 手动下载到跟这个脚本同一级的文件夹中. 然后跳到下一节代码中去.
注意这里一共下载了两个数据包,一个是`notMNIST_large.tar.gz`,作为训练集和验证集, 一个是`notMNIST_small.tar.gz`, 作为测试集.
``` python
# url = 'http://commondatastorage.googleapis.com/books1000/'
# 上面的url我打不开,百度了另外一个数据仓库如下
url = 'http://yaroslavvb.com/upload/notMNIST/'
last_percent_reported = None
data_root = '.'
notMNIST_large_size = 247336696
notMNIST_small_size = 8458043

def download_progress_hook(count, blockSize, totalSize):
    """ A hook to report the a
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(">")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download( filename, expected_bytes, force=False):
    """ helping:
        下面我们再来看看 urllib 模块提供的 urlretrieve() 函数。urlretrieve() 方法直接将远程数据下载到本地。

        1 >>> help(urllib.urlretrieve)
        2     Help on function urlretrieve in module urllib:

        urlretrieve(url, filename=None, reporthook=None, data=None)
        参数 finename 指定了保存本地路径（如果参数未指定，urllib会生成一个临时文件保存数据。）
        参数 reporthook 是一个回调函数，当连接上服务器、以及相应的数据块传输完毕时会触发该回调，我们可以利用这个回调函数来显示当前的下载进度。
        参数 data 指 post 到服务器的数据

        该方法返回一个包含两个元素的(filename, headers)元组，filename 表示保存到本地的路径，header 表示服务器的响应头。
    """
    dest_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(dest_filename):
        print("Attempint to download:", filename)
        filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        print("\nDownload Complete!")
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print("Found and verified", dest_filename)
    else:
        raise Exception(
                "Faild to verify " + dest_filename + ". Can you get to it with a browser?")
    return dest_filename

train_filename = maybe_download('notMNIST_large.tar.gz', notMNIST_large_size)
test_filename = maybe_download('notMNIST_small.tar.gz', notMNIST_small_size)

```

在下载的数据的时候,视频中给出的链接(也就是在代码中被注释掉的那个url)我下载不了,在浏览器中打开显示如下![Access-denied](Access-denied.png), 解决方法是使用上面代码中使用的url.
## 解压文件
``` python
num_classes = 10      # 共有十个文件夹, A到J
np.random.seed(133)

def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        print("%s already present - skipping extraction of %s." % (root, filename))
    else:
        print("Extracting data for %s. This may take a while, Please wait" % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    # 这是一段超级经典的保存文件路径的代码
    data_folders = [
            os.path.join(root, d)
            for d in sorted(os.listdir(root))
                if os.path.isdir(os.path.join(root, d))
                   ]
    if len(data_folders) != num_classes:
        raise Exception(
                "Expected %d folders, one per class. Found %d instead." % (num_classes, len(data_folders))
                )
    print(data_folders)
    return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
```

## 读取数据包中的每一个图像
``` python
image_size = 28
pixel_depth = 255.0

def load_letter(folder, min_num_images):
    """ Load the data for a single letter label. """
    image_files = os.listdir(folder)
    # 将每一个文件夹里面的图片都用一个大大的矩阵来保存起来
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32) 

    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2 ) / pixel_depth # 归一化处理
            if image_data.shape != (image_size, image_size):
                raise Exception("Unexpected image shape: %s" % str(image_data.shape) )
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, "- it\'s ok, skipping.")

    dataset = dataset[0 : num_images, :, :]
    if num_images < min_num_images:
        raise Exception("Many fewer images than expected: %d < %d " % (num_images, min_num_images))

    print("FULL dataset tensor:", dataset.shape)
    print("Mean:", np.mean(dataset))
    print("Standard deviation:", np.std(dataset))
    return dataset
```
然后,将该图像中的数据对象使用pickle函数保存为字节码文件,方便使用.
需要注意的是,pickle生成的字节码文件虽然很方便,但是存在安全隐患,熟悉python pickle对象构造原理的黑客,可以在该对象中植入任何的代码,让你的电脑执行,所以我们要使用可信任来源的pickle数据.
``` python
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force = True.
            print("%s already present - Skipping pickling." % set_filename)
        else:
            print("Pickling %s." % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print("Unable to save data to", set_filename, ':', e)
    return dataset_names

train_datasets = maybe_pickle( train_folders, 45000)
test_datasets = maybe_pickle( test_folders, 1800)

print(train_datasets)
print(test_datasets)

```
这一个过程耗费的时间比较长,我的机械硬盘大概需要40分钟的样子才能完成这一个过程.
## 还原几幅字节码文件中内容,确保数据仍旧完好
``` python
def display_sample_img( datasets, *, sample_size = 5, title = None ):
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title, fontsize = 16, fontweight = "bold")

    class_num = len( datasets )

    location = 1
    for dataset in datasets:
        with open( dataset, "rb") as f:
            sub_dataset = pickle.load(f)

            permutation = np.random.permutation( sample_size )
            sub_dataset = sub_dataset[permutation]

            for i in range( sample_size ):
                plt.subplot( class_num, sample_size, i+location )
                plt.imshow( sub_dataset[i] )
            location += sample_size

    plt.show()
    return

display_sample_img(train_datasets, sample_size = 10 )
display_sample_img(test_datasets, title = "test dataset samples")
```
## 数据重组
将10个文件夹中的数据合并到一起,并且进行重新洗牌,使每个手写字母数据充分混合,均匀分布

``` python
def make_arrays( nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray( (nb_rows, img_size, img_size), dtype = np.float32)
        labels = np.ndarray( nb_rows, dtype = np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets( pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays( valid_size, image_size)
    train_dataset, train_labels = make_arrays( train_size, image_size)

    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    # start_v == start of vaild dataset
    # start_t == start of train dataset
    # start_l == start of label set
    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validattiong and training set
                # 打乱数据集的顺序
                np.random.shuffle( letter_set )
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v : end_v, :, :] = valid_letter
                    valid_labels[start_v : end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                # 重上面获取的验证集之后的数据来获取训练集
                train_letter = letter_set[vsize_per_class : end_l, :, :]
                train_dataset[start_t : end_t, :, :] = train_letter
                train_labels[start_t : end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print("Unable to process data from", pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
        train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print("Training : ", train_dataset.shape, train_labels.shape)
print("Validation : ", valid_dataset.shape, valid_labels.shape)
print("Testing : ", test_dataset.shape, test_labels.shape)
```
## 最后的整理
将数据整理成一个完好的数据,来给接下来的任务进行使用.
``` python
def randomize( dataset, labels ):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize( train_dataset, train_labels)
test_dataset, test_labels = randomize( test_dataset, test_labels)
valid_dataset, valid_labels = randomize( valid_dataset, valid_labels)

picke_file = os.path.join( data_root, "notMNIST.pickle")

try:
    f = open( picke_file, "wb")
    save = {
            "train_dataset" : train_dataset,
            "train_labels"  : train_labels,
            "valid_dataset" : valid_dataset,
            "valid_labels"  : valid_labels,
            "test_dataset"  : test_dataset,
            "test_labels"   : test_labels,
            }
    pickle.dump( save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print("Unable to save data to", picke_file, ':', e)
    raise

statinfo = os.stat(picke_file)
print("Compressed picke size: ", statinfo.st_size)

```

## 训练一个简单的神经网络测试结果
``` python
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
def predict_accuracy( dataset , sample_size ):
    lgrg = LogisticRegression()
    with open(dataset, 'rb') as f:
        dataset = pickle.load(f)
        X_test = dataset["test_dataset"].reshape( dataset["test_dataset"].shape[0], 28*28)
        y_test = dataset["test_labels"]

        X_train = dataset["train_dataset"][:sample_size].reshape(sample_size, 784)
        y_train = dataset["train_labels"][:sample_size]

        lgrg.fit(X_train, y_train)

        print("\n sample size is ", sample_size, ", accuracy is ", lgrg.score(X_test, y_test))

predict_accuracy( picke_file, 100)
predict_accuracy( picke_file, 1000)
predict_accuracy( picke_file, 10000)
predict_accuracy( picke_file, 200000)
```
