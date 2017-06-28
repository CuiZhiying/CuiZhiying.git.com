---
title: "Mysql学习笔记之入门杂谈"
data: 2017-02-18 12:12:12
tags: [Mysql, note]
---

### 环境  
正在学习imooc上的[与MySQL的零距离接触](http://www.imooc.com/learn/122 "慕课网")，留下一点笔记作为以供回顾    
在`ubuntu 16.04`下安装了以下版本的mysql
`mysql  Ver 14.14 Distrib 5.7.16, for Linux (x86_64) using  EditLine wrapper`  
### 登陆
```shell
mysql -u username -p password
```
例如
```shell
mysql -u root -p
```
然后，就会提示输入管理员root的密码的，登陆数据库系统   
以上是我们最常用的登陆命令，完整的命令列表如下   

参数     | 全称     | 描述  
:------ | :------- | :------  
`-D`    | `--database = name`   | 打开指定数据库
        | `--delimiter = name`  | 指定分隔符
`-h`    | `--host = name `      | 服务器名字
`-p`    | `--password[=name]`   | 密码
`-P`    | `--port = #(3306为默认端口号)`    | 端口号
        | `--prompt = name`     | 设置提示符
`-u`    | `--user = name `      | 用户名
`-V`    | `--version`           | 输出版本信息并且退出         


### 常用命令
#### `Prompt`
进入到字符交互界面后，修改的提示符为`prompt`，其后的参数列表如下：

参数 | 描述
----|----
`\D`  | 完整的日期
`\d`  | 当前数据库
`\h`  | 服务器名称
`\u`  | 当前用户

例如：
```shell
Mysql> Prompt \u@\h \d> 
root@localhost (none)> 
```
#### `Select`  

命令                | 含义 
----               | ----
`SELECT VERSION（）`| 显示版本号
`SELECT NOW（）`    | 显示当前日期时间
`SELECT NSER（）`   | 显示当前用户  

### 语句规范
- 关键字与函数名称全部大写
- 数据库名称、表名称、字段名称全部小写
- SQL语句必须以分号结尾
