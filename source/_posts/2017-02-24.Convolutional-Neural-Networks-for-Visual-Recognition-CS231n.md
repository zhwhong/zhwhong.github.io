---
title: '[斯坦福CS231n课程整理] Convolutional Neural Networks for Visual Recognition(附翻译，作业)'
date: 2017-02-24 17:27:52
tags: [Deep Learning, CNN, Computer Vision]
categories: Machine Learning
---

> # CS231n课程：面向视觉识别的卷积神经网络

- 课程官网：[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- Github：https://github.com/cs231n/cs231n.github.io | http://cs231n.github.io/
- 教学安排及大纲：[Schedule and Syllabus](http://vision.stanford.edu/teaching/cs231n/syllabus.html)
- 课程视频：Youtube上查看[Andrej Karpathy](https://link.zhihu.com/?target=https%3A//www.youtube.com/channel/UCPk8m_r6fkUSYmvgCBwq-sw)创建的[播放列表](https://link.zhihu.com/?target=https%3A//www.youtube.com/playlist%3Flist%3DPLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)，或者[网易云课堂](http://study.163.com/course/introduction/1003223001.htm#/courseDetail)
- 课程pdf及视频下载：[百度网盘下载](https://pan.baidu.com/s/1eRHH4L8)，密码是4efx

<!--more-->

![](http://upload-images.jianshu.io/upload_images/145616-a0eeadfcd667b7bb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> # 授课 ([Stanford Vision Lab](http://vision.stanford.edu/index.html))

- [Fei-Fei Li](http://vision.stanford.edu/feifeili/) (Associate Professor, Stanford University)
- [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/) | [Github](https://github.com/karpathy) | [Blog](http://karpathy.github.io/) | [Twitter](https://twitter.com/karpathy)
- [Justin Johnson](http://cs.stanford.edu/people/jcjohns/) | [Github](https://github.com/jcjohnson)

![Course Instructors and Teaching Assistants](http://upload-images.jianshu.io/upload_images/145616-df9eb2f6ea9512fb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> # 课程原文 & 作业 & 中文翻译笔记
> - 知乎专栏：[**智能单元**](https://zhuanlan.zhihu.com/intelligentunit)
> - 作者：[**杜客**](https://www.zhihu.com/people/du-ke/answers) (在此对作者表示特别感谢！)

![翻译得到Karpathy许可](http://upload-images.jianshu.io/upload_images/145616-3b415a85af702e04.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- [贺完结！CS231n官方笔记授权翻译总集篇发布](https://zhuanlan.zhihu.com/p/21930884?refer=intelligentunit)
- [获得授权翻译斯坦福CS231n课程笔记系列](https://zhuanlan.zhihu.com/p/20870307?refer=intelligentunit)
- [CS231n课程笔记翻译：Python Numpy教程](https://zhuanlan.zhihu.com/p/20878530?refer=intelligentunit) | [课程原文](http://cs231n.github.io/python-numpy-tutorial/)
- [CS231n课程笔记翻译：图像分类笔记（上）](https://zhuanlan.zhihu.com/p/20894041?refer=intelligentunit) | [课程原文](http://cs231n.github.io/classification/)
- [CS231n课程笔记翻译：图像分类笔记（下）](https://zhuanlan.zhihu.com/p/20900216?refer=intelligentunit)
- [CS231n课程笔记翻译：线性分类笔记（上）](https://zhuanlan.zhihu.com/p/20918580?refer=intelligentunit) | [课程原文](http://cs231n.github.io/linear-classify/)
- [CS231n课程笔记翻译：线性分类笔记（中）](https://zhuanlan.zhihu.com/p/20945670?refer=intelligentunit)
- [CS231n课程笔记翻译：线性分类笔记（下）](https://zhuanlan.zhihu.com/p/21102293?refer=intelligentunit)
- [知友智靖远关于CS231n课程字幕翻译的倡议 ](https://zhuanlan.zhihu.com/p/21354230?refer=intelligentunit)
- [CS231n课程笔记翻译：最优化笔记（上）](https://zhuanlan.zhihu.com/p/21360434?refer=intelligentunit) | [课程原文](http://cs231n.github.io/optimization-1/)
- [CS231n课程笔记翻译：最优化笔记（下）](https://zhuanlan.zhihu.com/p/21387326?refer=intelligentunit)
- [CS231n课程笔记翻译：反向传播笔记 ](https://zhuanlan.zhihu.com/p/21407711?refer=intelligentunit) | [课程原文](http://cs231n.github.io/optimization-2/)
- [斯坦福CS231n课程作业 # 1 简介 ](https://zhuanlan.zhihu.com/p/21441838?refer=intelligentunit) | [课程原文](http://cs231n.github.io/assignments2016/assignment1/)
- [CS231n课程笔记翻译：神经网络笔记 1（上）](https://zhuanlan.zhihu.com/p/21462488?refer=intelligentunit) | [课程原文](http://cs231n.github.io/neural-networks-1/)
- [CS231n课程笔记翻译：神经网络笔记 1（下）](https://zhuanlan.zhihu.com/p/21513367?refer=intelligentunit)
- [CS231n课程笔记翻译：神经网络笔记 2 ](https://zhuanlan.zhihu.com/p/21560667?refer=intelligentunit) | [课程原文](http://cs231n.github.io/neural-networks-2/)
- [CS231n课程笔记翻译：神经网络笔记 3（上）](https://zhuanlan.zhihu.com/p/21741716?refer=intelligentunit) | [课程原文](http://cs231n.github.io/neural-networks-3/)
- [CS231n课程笔记翻译：神经网络笔记 3（下）](https://zhuanlan.zhihu.com/p/21798784?refer=intelligentunit)
- [斯坦福CS231n课程作业 # 2 简介 ](https://zhuanlan.zhihu.com/p/21941485?refer=intelligentunit) | [课程原文](http://cs231n.github.io/assignments2016/assignment2/)
- [CS231n课程笔记翻译：卷积神经网络笔记 ](https://zhuanlan.zhihu.com/p/22038289?refer=intelligentunit) | [课程原文](http://cs231n.github.io/convolutional-networks/)
- [斯坦福CS231n课程作业 # 3 简介 ](https://zhuanlan.zhihu.com/p/21946525?refer=intelligentunit) | [课程原文](http://cs231n.github.io/assignments2016/assignment3/)
- [Andrej Karpathy的回信和Quora活动邀请](https://zhuanlan.zhihu.com/p/22282421?refer=intelligentunit)
- [知行合一码作业，深度学习真入门 ](https://zhuanlan.zhihu.com/p/22232836?refer=intelligentunit)


**【附录 - Assignment】：**

- [简书] [CS231n (winter 2016) : Assignment1](http://www.jianshu.com/p/004c99623104)
- [简书] [CS231n (winter 2016) : Assignment2](http://www.jianshu.com/p/9c4396653324)
- [简书] [CS231n (winter 2016) : Assignment3（更新中）](http://www.jianshu.com/p/e46b1aa48886)
- [Github] CS231n作业[ 参考1](https://github.com/MyHumbleSelf/cs231n) | [参考2](https://github.com/dengfy/cs231n) ……


---

(再次感谢[智能单元-知乎专栏](https://zhuanlan.zhihu.com/intelligentunit)，以及知乎作者[@杜客](https://www.zhihu.com/people/du-ke/answers)和相关朋友[@ShiqingFan](https://www.zhihu.com/people/584f06e4ed2edc6007e4793179e7cdc1)，[@猴子](https://www.zhihu.com/people/hmonkey)，[@堃堃](https://www.zhihu.com/people/e7fcc05b0cf8a90a3e676d0206f888c9)，[@李艺颖](https://www.zhihu.com/people/f11e78650e8185db2b013af42fd9a481)等为CS231n课程翻译工作做出的贡献，辛苦了！)

---

**其他课程整理：**

- [[斯坦福CS224d课程整理] Natural Language Processing with Deep Learning](http://www.jianshu.com/p/062d2bbbef93) @ [zhwhong](http://www.jianshu.com/u/38cd2a8c425e)
- [[斯坦福CS229课程整理] Machine Learning Autumn 2016](http://www.jianshu.com/p/0a6ef31ff77a) @ [zhwhong](http://www.jianshu.com/u/38cd2a8c425e)
- [[coursera 机器学习课程] Machine Learning by Andrew Ng](http://www.jianshu.com/p/c68d0df13e0b) @ [zhwhong](http://www.jianshu.com/u/38cd2a8c425e)
