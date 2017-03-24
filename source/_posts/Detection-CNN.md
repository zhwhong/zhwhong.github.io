---
title: '[Detection] CNN 之 "物体检测" 篇'
date: 2017-02-24 18:20:50
tags: [Object Detection]
categories: Machine Learning
---

# Index
- [RCNN](http://arxiv.org/abs/1311.2524) 
- [Fast RCNN](http://arxiv.org/abs/1504.08083) 
- [Faster RCNN](http://arxiv.org/abs/1506.01497) 
- [R-FCN](http://arxiv.org/pdf/1605.06409v1.pdf)
- [YOLO](http://arxiv.org/abs/1506.02640) 
- [SSD](http://www.cs.unc.edu/~wliu/papers/ssd.pdf) 
- NMS
- xywh VS xyxy

---

<!--more-->

# RCNN
[Rich feature hierarchies for accurate object detection and semantic segmentation](http://arxiv.org/abs/1311.2524) 

早期，使用窗口扫描进行物体识别，计算量大。 RCNN去掉窗口扫描，用聚类方式，对图像进行分割分组，得到多个侯选框的层次组。

![](http://upload-images.jianshu.io/upload_images/145616-f4c5c9a89c842dcb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 原始图片通过Selective Search提取候选框，约有2k个
- 侯选框缩放成固定大小
- 经过CNN
- 经两个全连接后，分类

> 拓展阅读：[基于R-CNN的物体检测-CVPR 2014](http://blog.csdn.net/hjimce/article/details/50187029)

# Fast RCNN
[Fast R-CNN](http://arxiv.org/abs/1504.08083) 

RCNN中有CNN重复计算，Fast RCNN则去掉重复计算，并微调选框位置。

![](http://upload-images.jianshu.io/upload_images/145616-1d610559358abecf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 整图经过CNN，得到特征图
- 提取域候选框
- 把候选框投影到特征图上，Pooling采样成固定大小
- 经两个全连接后，分类与微调选框位置

# Faster RCNN
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/abs/1506.01497) 

提取候选框运行在CPU上，耗时2s，效率低下。 
Faster RCNN使用CNN来预测候选框。

![](http://upload-images.jianshu.io/upload_images/145616-8b71602ad793eed9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 整图经过CNN，得到特征图
- 经过核为 3×3×256 的卷积，每个点上预测k个anchor box是否是物体，并微调anchor box的位置
- 提取出物体框后，采用Fast RCNN同样的方式，进行分类
- 选框与分类共用一个CNN网络

anchor box的设置应比较好的覆盖到不同大小区域，如下图:

![](http://upload-images.jianshu.io/upload_images/145616-833ff24cf66a7fd6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

一张1000×600的图片，大概可以得到20k个anchor box(60×40×9)。

# R-FCN
[R-FCN: Object Detection via Region-based Fully Convolutional Networks](http://arxiv.org/pdf/1605.06409v1.pdf) 

> 论文翻译详见：[[译] 基于R-FCN的物体检测 (zhwhong)](http://www.jianshu.com/p/db1b74770e52)

RCNN系列(RCNN、Fast RCNN、Faster RCNN)中，网络由两个子CNN构成。在图片分类中，只需一个CNN，效率非常高。所以物体检测是不是也可以只用一个CNN？ 

图片分类需要兼容形变，而物体检测需要利用形变，如何平衡？ 

R-FCN利用在CNN的最后进行位置相关的特征pooling来解决以上两个问题。

![](http://upload-images.jianshu.io/upload_images/145616-8eb1556488b4fdc2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

经普通CNN后，做有 k^2(C+1) 个 channel 的卷积，生成位置相关的特征(position-sensitive score maps)。

C 表示分类数，加 1 表示背景，k 表示后续要pooling 的大小，所以生成 k^2 倍的channel，以应对后面的空间pooling。

![](http://upload-images.jianshu.io/upload_images/145616-0bcb1e46be5e24c5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

普通CNN后，还有一个RPN(Region Proposal Network)，生成候选框。

假设一个候选框大小为 w×h，将它投影在位置相关的特征上，并采用average-pooling的方式生成一个 k×k×k^2(C+1) 的块(与Fast RCNN一样)，再采用空间相关的pooling(k×k平面上每一个点取channel上对应的部分数据)，生成 k×k×(C+1)的块，最后再做average-pooling生成 C+1 的块，最后做softmax生成分类概率。

类似的，RPN也可以采用空间pooling的结构，生成一个channel为 4k^2的特征层。

空间pooling的具体操作可以参考下面。

![](http://upload-images.jianshu.io/upload_images/145616-4411b2baa05764f0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

训练与SSD相似，训练时拿来做lost计算的点取一个常数，如128。 除去正点，剩下的所有使用概率最高的负点。

# YOLO
[You Only Look Once: Unified, Real-Time Object Detection](http://arxiv.org/abs/1506.02640) 

Faster RCNN需要对20k个anchor box进行判断是否是物体，然后再进行物体识别，分成了两步。 YOLO则把物体框的选择与识别进行了结合，一步输出，即变成”You Only Look Once”。

![](http://upload-images.jianshu.io/upload_images/145616-881c58173e5fab4b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 把原始图片缩放成448×448大小
- 运行单个CNN
- 计算物体中心是否落入单元格、物体的位置、物体的类别

模型如下:

![](http://upload-images.jianshu.io/upload_images/145616-148936c1f19644a4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 把缩放成统一大小的图片分割成S×S的单元格
- 每个单元格输出B个矩形框(冗余设计)，包含框的位置信息(x, y, w, h)与物体的Confidence
- 每个单元格再输出C个类别的条件概率P(Class∣Object)
- 最终输出层应有S×S×(B∗5+C)个单元
- x, y 是每个单元格的相对位置
- w, h 是整图的相对大小

Conficence定义如下:

![](http://upload-images.jianshu.io/upload_images/145616-772c5abc28591971.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在原论文中，S = 7，B = 2，C = 20，所以输出的单元数为7×7×30。

![](http://upload-images.jianshu.io/upload_images/145616-a8d08b9a46de7f9f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

代价函数：

![](http://upload-images.jianshu.io/upload_images/145616-7f2b9e54c3730d5f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中 `λ_coord=5`，`λ_noobj=0.5`。
一般，w与 h 不是在 [0,1]上的均匀分布，偏小，所以开方。

**注: 开方的解释是我自己的估计，可能不对。**

# SSD

[SSD: Single Shot MultiBox Detector](http://www.cs.unc.edu/~wliu/papers/ssd.pdf) 

YOLO在 7×7 的框架下识别物体，遇到大量小物体时，难以处理。
SSD则在不同层级的feature map下进行识别，能够覆盖更多范围。

![](http://upload-images.jianshu.io/upload_images/145616-f481d203b810dba3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

假设在 m 层 feature map 上进行识别，则第 k 层的基本比例为

![](http://upload-images.jianshu.io/upload_images/145616-e1ea76b11f39c48a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

比如 s_min=0.2，s_max=0.95，表示整张图片识别物体所占比最小 0.2，最大 0.95。

在基本比例上，再取多个长宽比，令 a={1, 2, 3, 1/2, 1/3}，长宽分别为

![](http://upload-images.jianshu.io/upload_images/145616-d79012d45d6f57a8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Match策略上，取ground truth与以上生成的格子重叠率大于0.5的。

# SSD vs YOLO

![](http://upload-images.jianshu.io/upload_images/145616-70f2fd38db66b76e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

位置采用Smooth L1 Regression，分类采用Softmax。 
代价函数为：

![][01]
[01]:http://latex.codecogs.com/png.latex?L%20=%20L_{conf}(x,%20c)%20+%20\alpha%20\cdot%20L_{loc}(c,%20l,%20g))

x  表示类别输出，c 表示目标分类，l 表示位置输出，g 表示目标位置, α是比例常数，可取1。
训练过程中负点远多于正点，所以只取负点中，概率最大的几个，数量与正点成 3:1 。

# NMS
以上方法，同一物体可能有多个预测值。 
可用NMS(Non-maximum suppression，非极大值抑制)来去重。

![](http://upload-images.jianshu.io/upload_images/145616-ba89e4a3fde65974.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如上图所示，一共有6个识别为人的框，每一个框有一个置信率。 
现在需要消除多余的:
- 按置信率排序: 0.95, 0.9, 0.9, 0.8, 0.7, 0.7
- 取最大0.95的框为一个物体框
- 剩余5个框中，去掉与0.95框重叠率大于0.6(可以另行设置)，则保留0.9, 0.8, 0.7三个框
- 重复上面的步骤，直到没有框了，0.9为一个框
- 选出来的为: 0.95, 0.9

两个矩形的重叠率计算方式如下:

![](http://upload-images.jianshu.io/upload_images/145616-59ba4b17d2cc2538.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# xywh VS xyxy

系列论文中，位置都用 (x,y,w,h)来表示，没有用左上角、右下角 (x,y,x,y) 来表示。
初衷是当 (w,h)正确时，(x,y) 一点错，会导致整个框就不准了。
在初步的实际实验中，(x,y,x,y) 效果要差一些。

背后的逻辑，物体位置用 (x,y,w,h) 来学习比较容易。
(x,y) 只需要位置相关的加权就能计算出来；
(w,h) 就更简单了，直接特征值相加即可。

---

- 原文链接：[Detection](http://www.cosmosshadow.com/ml/%E5%BA%94%E7%94%A8/2015/12/07/%E7%89%A9%E4%BD%93%E6%A3%80%E6%B5%8B.html)
- 参考：[[译] 基于R-FCN的物体检测 (zhwhong)](http://www.jianshu.com/p/db1b74770e52)

(转载请联系作者并注明出处，谢谢！)
