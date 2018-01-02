---
title: git clone fail
date: 2017-12-12 14:14:26
categories: [errors, bug, proxy]
tags: [ss, fj]
---

## 起因
墙外的世界很精彩,之前使用lantern翻墙,今年9月分lantern挂了一段时间,在这段时间里,卸载了lantern,改用ss.
很久了,突然发现我用`git clone https://github.com/who/repo.git` 的命令的时候 报出一个错误
```
fatal: unable to access 'https://github.com/sublime-emacs/sublemacspro.git/': Failed to receive SOCKS4 connect request ack.
```

就再也没办法愉快地从github上下载代码了,同样的,使用Vunble管理的vim插件也更新不了了,查看记录,发现原因同上.

## 经过
赶紧Google一下,[_**推荐回答**_](https://segmentfault.com/q/1010000003536027/a-1020000003569552)是代理出问题了,说代理挂掉了,可是我明明可以在浏览器上愉快得玩耍呀.
其他的回答更加不靠谱,我也没看明白.大概发现了是curl程序下载出错了.
最后鬼使神差,让我看到了这个[_**帖子**_](https://gist.github.com/laispace/666dd7b27e9116faece6).才知道,原来使用了ss之后,git要单独设置代理.

## 结果
git设置代理的方法如下:
``` bash
~$ git config --global http.proxy "socks5://127.0.0.1:1080"
~$ git config --global https.proxy "socks5://127.0.0.1:1080"
```
取消设置的命令如下:
``` bash
~$ git config --global --unset http.proxy
~$ git config --global --unset https.proxy
```
git查看全局配置信息的命令如下:
``` bash
git config --list
```
从结果中可以看到自己的信息类似如下:
``` bash
david@potato:~$ git config --list
user.email=cuizhiying.csu@gmail.com
user.name=david
https.proxy=socks5://127.0.0.1:1080
http.proxy=socks5://127.0.0.1:1080
```
