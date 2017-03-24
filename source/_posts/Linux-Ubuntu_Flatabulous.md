---
title: '[Linux] Ubuntu下超好看扁平主题 : Flatabulous'
date: 2017-02-24 16:12:16
tags: [Ubuntu, Theme]
categories: Linux
---

![Flatabulous](http://upload-images.jianshu.io/upload_images/145616-909d61913233d890.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

<!--more-->

使用ubuntu的小伙伴们，不知道你们对ubuntu自带主题有什么看法，反正我个人不太喜欢，个人比较喜欢扁平化的风格。
下面给大家推荐一个我长期使用的扁平化风格的主题－[Flatabulous](https://github.com/anmoljagetia/Flatabulous) 。
先看一下我的桌面(个人比较偏向单色调，不要在意这些细节啦)：

![My Desktop](http://upload-images.jianshu.io/upload_images/145616-1564d71f915f7cef.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

那么**Flatabulous**到底是什么呢？
　　"This is a Flat theme for Ubuntu and other debian based Linux Systems. This is based on the Ultra-Flat theme. Special thanks to @steftrikia and Satyajit Sahoo for the original work."
哈哈，不卖关子了，它其实就是一个超级好看的扁平化Ubuntu主题。

![](http://upload-images.jianshu.io/upload_images/145616-e639fe182b0b743b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面就开始说说怎么安装它吧~

# [ 安装 ]

## Step 1　安装 Unity Tweak Tool

要安装这个主题，首先要安装[Unity Tweak Tool](https://launchpad.net/unity-tweak-tool)或者[Ubuntu Tweak Tool](https://github.com/tualatrix/ubuntu-tweak)。
安装Unity Tweak Tool可以很简单地执行如下命令：
```
$ sudo apt-get install unity-tweak-tool
```
安装Ubuntu Tweak Tool可以使用如下命令：
```
$ sudo add-apt-repository ppa:tualatrix/ppa  
$ sudo apt-get update
$ sudo apt-get install ubuntu-tweak
```
或者跑到它们的网站下载.deb文件(推荐)，打开Ubuntu软件中心安装或者使用命令`dpkg -i`(推荐)安装。

注：If you are on Ubuntu 16.04 or higher, run the commands below to install Ubuntu Tweak:

```
$ wget -q -O - http://archive.getdeb.net/getdeb-archive.key | sudo apt-key add -
$ sudo sh -c 'echo "deb http://archive.getdeb.net/ubuntu xenial-getdeb apps" >> /etc/apt/sources.list.d/getdeb.list'
$ sudo apt-get update
$ sudo apt-get install ubuntu-tweak
```

安装完毕后，我们可以就搜到Ubuntu Tweak这款软件了，如下图：

![](http://upload-images.jianshu.io/upload_images/145616-c073230df8a73b8b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## Step 2　安装Flatabulous主题

### 方式1：Using the .deb file for Debian, Ubuntu and derivatives (Recommended)

下载.deb文件，点击[这里](https://github.com/anmoljagetia/Flatabulous/releases)，下载后，打开Ubuntu软件中心或者使用命令`dpkg -i`（推荐）安装。

### 方式2：Using the noobslab PPA

```
$ sudo add-apt-repository ppa:noobslab/themes
$ sudo apt-get update
$ sudo apt-get install flatabulous-theme
```

### 方式3：下载Flatabulous源码

下载主题源码，点击[这里](https://github.com/anmoljagetia/Flatabulous/archive/master.zip)，或者使用git克隆下来，Github仓库地址： [https://github.com/anmoljagetia/Flatabulous](https://github.com/anmoljagetia/Flatabulous)
如果下载的是zip文件，先将其解压，然后移动到/usr/share/themes/下。如果是git clone下来的，直接执行下如下命令：

```
$ sudo mv Flatabulous /usr/share/themes/
```

## Step 3　Tweak配置

我们打开Ubuntu Tweak，选择**调整->主题**，如下：

![](http://upload-images.jianshu.io/upload_images/145616-c937f438f034d8bd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

然后，配置GTK主题和窗口主题，选择Flatabulous，如下：

![](http://upload-images.jianshu.io/upload_images/145616-1e90b53fc9d14f68.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

你们可以模仿我的配置，不过此时还有一个问题，就是你发现图标主题没有`Ultra-Flat`选项，这个`icon`需要额外下载，原生的`Tweak`里面并没有。
对于图标，我使用的是ultra-flat-icons主题。有蓝色（推荐），橙色和薄荷绿颜色可用。要安装它，你可以运行下面的命令：
```
$ sudo add-apt-repository ppa:noobslab/icons
$ sudo apt-get update
$ sudo apt-get install ultra-flat-icons
```
或者你也可以运行`sudo apt-get install ultra-flat-icons-orange`或者 `sudo apt-get install ultra-flat-icons-green`。
根据你自己喜欢的颜色选择，我推荐的是扁平图标，但是你也可以看看**Numix**和**Flattr**。

图标安装好后，再打开Ubuntu Tweak，选择 `调整->主题`，选择图标主题为`Ultra-Flat`。

安装完以后，只需要在theme进行相应的配置，然后换一个自己喜欢的桌面壁纸，我们就能看到超级好看的ubuntu啦。如果不行，重启计算机，应该就可以了。重启之后你的计算机看起来差不多是这样的：



![扁平化图标](http://upload-images.jianshu.io/upload_images/145616-a9408b3132214304.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


# [ 部分效果图截图 ]


## 文件管理

![](http://upload-images.jianshu.io/upload_images/145616-63d8986b44433f99.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## Theme with Sublime Text 3 and JavaScript Code

![](http://upload-images.jianshu.io/upload_images/145616-ac6eca94b5b50a2e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 系统设置

![](http://upload-images.jianshu.io/upload_images/145616-b0e714419dcd6433.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## Posters

![](http://upload-images.jianshu.io/upload_images/145616-9c99a2b56e70c30f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](http://upload-images.jianshu.io/upload_images/145616-0e6a006a57adb9d3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## Terminal

![](http://upload-images.jianshu.io/upload_images/145616-421d8c2880c84c62.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# [ Reference ]

- [Flatabulous：超级好看的Ubuntu 扁平主题](http://www.xulukun.cn/flatabulous-ubuntu.html)
- [Github -> Flatabulous](https://github.com/anmoljagetia/Flatabulous)

(转载请注明原作者及出处, 谢谢！)
