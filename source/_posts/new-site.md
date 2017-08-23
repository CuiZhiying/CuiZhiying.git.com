---
title: 新的站点Hexo + NexT
date: 2017-06-27 01:29:43
tags:
categories: this_site
---

## 前言
开了Github很久，一直都在做代码的搬运工，后发现，很多东西自己做了，又忘了，要用的时候，又得重新百度资料。想想，不如在学东西的自己将资料稍微整理一下，既加深了理解，有方便日后的总结。

## Hexo
之前使用的用过[Jekyll]()，但是实在觉得有点复杂，hexo是我觉得最简单的一个了，而且有些小清新的主题，果断入手。

1. [廖雪峰的git教程](http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/)
2. [令狐葱的建站教程](https://linghucong.js.org/2016/04/15/2016-04-15-hexo-github-pages-blog/)
3. [Hexo中文文档](https://hexo.io/zh-cn/docs/)
4. [NexT主题官方文档](http://theme-next.iissnan.com/getting-started.html)

### github page
首先，我是通过廖雪峰的网站来入门git以及搭建github仓库的，有了这部分的基础，搭建github page会容易很多   
然后就是看到了别人的主页，以及很多的教程，我做过很多尝试，来理解转化引擎既jekyll的文件结构，那个文件目录比较复杂，理解也比较困难。然后就是看到了**[令狐葱](https://linghucong.js.org/)**大神的构建过程。没错，我的主页和模板跟他的都是一样的。：）
但是我觉得直接看官方文档的介绍会做的比较顺利和系统的一些，而令狐大神的介绍的话，则可以给我们提供一个实际操作的范例。
NexT主题的配置同样是如此。
### 我的踩坑介绍
- 主页上的Categories的按钮和Tags的按钮点击之后，显示找不到页面。
  那是因为没有在Hexo的文件目录中建立起相应的文件。具体的操作可以参考NexT的官方文档。[传送门](http://theme-next.iissnan.com/theme-settings.html#tags-page)   
- 主页上的文件没有缩略，显示的全文内容
   在`hexo/themes/next/_config.yml`文件中，找到以下代码，并且将enable后面的设置改成`true`即可，`length`设置的是显示文章前多少个字符
   ```yml
   auto_excerpt:
      enable: true
      length: 150

   ```
## 结语
折腾了以下，这个博客应该是基本稳定了，以要专注内容的更新了！
