---
title: C指针初始化踩坑
date: 2017-08-23 14:40:31
categories:
tags: C-pointer
---
#### 正确演示
``` C
int *p;
int a = 25
p = &a;
*p = 20;
p = "abcdefg"
```
这里是指针先声明，后初始化。对`p`进行初始化时，要赋地址值，对`*p`进行初始化的时候要赋声明的数据类型，其中，`*`表示的是间接取值的意思。

或者在声明的同时初始化。
``` C
int a = 25;
int b[10] = {0};

int *p = &a;
int *p = b;
int *p = &b[0];
```
此时，`=`号右边的操作数必须为内存地址，`*p`只是表示这是一个指针变量，并没有间接取值的意思。   
#### 错误演示

``` C
int a = 25;

int *apple = 25;
int *set = {2, 3, 5};
int *number = a;
```

以上三种方法是错误的，我又犯了几次, 原因同上，`*`在这里只是声明该变量是一个指针，并没有间接取值的意思。
#### 终极错误
``` C
int *a;
*a = 1;
```
好吧，这是一个能反应C指针本质的错误。这种写法，编译没有报错，运行时会出现内存错误。
上面第一行只声明了一个指针变量，里面保存的是一个（32位）的内存地址，但是第二行并没有声明该指针变量保存的是哪一块内存的地址，就直接间接取值符号`*`来进行初始化了，这样子会让那个数字`1`无处安放，因为指针变量a还没有指向一块内存。
如果还没哟理解的话，可以看看下面的修改：
``` C
int *a;
int b;
a = &b;
*a = 1;
```
上面的代码声明了多一块内存空间`b`，`*a`指的是将数据`1`保存在`b`所声明的内存空间中。