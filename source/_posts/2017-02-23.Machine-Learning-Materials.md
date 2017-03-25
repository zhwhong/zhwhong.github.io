---
title: Machine Learning Materials
date: 2017-02-23 16:57:14
tags: [Deep Learning, TensorFlow, CNN, RNN, LSTM, Object Detection]
categories: Machine Learning
top: 2
description: "本篇文章整理、归纳了自己学习Deep Learning方面的一些资料，包括GitHub Awesome，DL框架如TensorFlow，分布式教程，卷积神经网络CNN，物体检测Paper，循环神经网络RNN、LSTM等，以及斯坦福CS231n计算机视觉识别和Coursera Andrew Ng机器学习等相关课程整理。"
---

![Machine Learning](machine_learning_materials.png)

---

## Awesome系列　

- [**Awesome Machine Learning**](https://github.com/josephmisiti/awesome-machine-learning)
- [**Awesome Deep Learning**](https://github.com/ChristosChristofidis/awesome-deep-learning)
- [**Awesome TensorFlow**](https://github.com/jtoy/awesome-tensorflow)
- [Awesome TensorFlow Implementations](https://github.com/TensorFlowKR/awesome_tensorflow_implementations)
- [Awesome Torch](https://github.com/carpedm20/awesome-torch)
- [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision)
- [Awesome Deep Vision](https://github.com/kjw0612/awesome-deep-vision)
- [Awesome RNN](https://github.com/kjw0612/awesome-rnn)
- [Awesome NLP](https://github.com/keonkim/awesome-nlp)
- [Awesome AI](https://github.com/owainlewis/awesome-artificial-intelligence)
- [Awesome Deep Learning Papers](https://github.com/terryum/awesome-deep-learning-papers)
- [Awesome 2vec](https://github.com/MaxwellRebo/awesome-2vec)


## Deep Learning

- [Book] [**Neural Networks and Deep Learning**](http://neuralnetworksanddeeplearning.com/chap1.html) 中文翻译(不完整): [神经网络与深度学习](https://www.gitbook.com/book/hit-scir/neural-networks-and-deep-learning-zh_cn/details) 第五章中文翻译: [[译] 第五章 深度神经网络为何很难训练](http://www.jianshu.com/p/917f71b06499)
- [Book] [Deep Learning - MIT Press](http://www.deeplearningbook.org/)
- [Book] [Pattern Recognition and Machine Learning](http://www.springer.com/gb/book/9780387310732) (Bishop) | [豆瓣](https://book.douban.com/subject/2061116/) | [PRML & DL笔记](http://nbviewer.jupyter.org/github/lijin-THU/notes-machine-learning/blob/master/ReadMe.ipynb) | [GitBook](https://www.gitbook.com/book/mqshen/prml/details)
- [Course] [**Deep Learning - Udacity**](https://cn.udacity.com/course/deep-learning--ud730/)
- [Course] [**Machine Learning by Andrew Ng - Coursera**](https://www.coursera.org/learn/machine-learning) | [**课程资料整理**](http://www.jianshu.com/p/c68d0df13e0b) @ [zhwhong](http://www.jianshu.com/u/38cd2a8c425e)
- [Course] [**Convolutional Neural Networks for Visual Recognition(CS231n)**](http://cs231n.stanford.edu/) | [**课程资料整理**](http://www.jianshu.com/p/182baeb82c71) @ [zhwhong](http://www.jianshu.com/u/38cd2a8c425e)
- [Course] [Deep Learning for Natural Language Processing(CS224d)](http://cs224d.stanford.edu/) | [课程资料整理](http://www.jianshu.com/p/062d2bbbef93) @ [zhwhong](http://www.jianshu.com/u/38cd2a8c425e)
- [View] [Top Deep Learning Projects on Github](https://github.com/aymericdamien/TopDeepLearning)
- [View] [Deep Learning for NLP resources](https://github.com/andrewt3000/DL4NLP/blob/master/README.md)
- [View] [资源 | 深度学习资料大全：从基础到各种网络模型](http://www.jianshu.com/p/6752a8845d01)
- [View] [Paper | DL相关论文中文翻译](http://www.jianshu.com/nb/8413272)
- [View] [深度学习新星：GAN的基本原理、应用和走向](http://www.jianshu.com/p/80bd4d4c2992)
- [View] [推荐 | 九本不容错过的深度学习和神经网络书籍](http://www.jianshu.com/p/c20917a91472)
- [View] [Github好东西传送门](https://github.com/memect/hao) --> [深度学习入门与综述资料](https://github.com/memect/hao/blob/master/awesome/deep-learning-introduction.md)

## Frameworks

- [TensorFlow (by google)](https://www.tensorflow.org/)
- [MXNet](https://github.com/dmlc/mxnet)
- [Torch (by Facebook)](http://torch.ch/)
- [Caffe (by UC Berkley)([http://caffe.berkeleyvision.org/](http://caffe.berkeleyvision.org/))
- [Deeplearning4j([http://deeplearning4j.org](http://deeplearning4j.org/))
- Brainstorm
- Theano、Chainer、Marvin、Neon、ConvNetJS

## TensorFlow

- 官方文档
- [TensorFlow Tutorial](https://www.tensorflow.org/tutorials)
- [TensorFlow 官方文档中文版](http://wiki.jikexueyuan.com/project/tensorflow-zh/)
- [TensorFlow Whitepaper](http://download.tensorflow.org/paper/whitepaper2015.pdf)
- [[译] TensorFlow白皮书](http://www.jianshu.com/p/65dc64e4c81f)
- [API] [API Document](https://www.tensorflow.org/versions/r0.8/api_docs/index.html)

## 入门教程

- [教程] [Learning TensorFlow](http://learningtensorflow.com/index.html)
- [TensorFlow-Tutorials @ github](https://github.com/nlintz/TensorFlow-Tutorials) (推荐)
- [Awesome-TensorFlow](https://github.com/jtoy/awesome-tensorflow) (推荐)
- [TensorFlow-Examples @ github](https://github.com/aymericdamien/TensorFlow-Examples)
- [tensorflow_tutorials @ github](https://github.com/pkmital/tensorflow_tutorials)

## 分布式教程

- [Distributed TensorFlow官方文档](https://www.tensorflow.org/versions/r0.8/how_tos/distributed/index.html#distributed-tensorflow)
- [distributed-tensorflow-example @ github](https://github.com/ischlag/distributed-tensorflow-example) (推荐)
- [DistributedTensorFlowSample @ github](https://github.com/ashitani/DistributedTensorFlowSample)
- [Parameter Server](http://parameterserver.org/)

## Paper (Model)

### CNN Nets

- [LeNet](http://yann.lecun.com/exdb/lenet/)
- [AlexNet](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)
- [OverFeat](https://arxiv.org/abs/1312.6229v4)
- [NIN](https://arxiv.org/abs/1312.4400v3)
- [GoogLeNet](http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)
- [Inception-V1](https://arxiv.org/abs/1409.4842v1)
- [Inception-V2](https://arxiv.org/abs/1502.03167)
- [Inception-V3](http://arxiv.org/abs/1512.00567)
- [Inception-V4](https://arxiv.org/abs/1602.07261)
- [Inception-ResNet-v2](http://arxiv.org/abs/1602.07261)
- [ResNet 50](https://arxiv.org/abs/1512.03385)
- [ResNet 101](https://arxiv.org/abs/1512.03385)
- [ResNet 152](https://arxiv.org/abs/1512.03385)
- [VGG 16](http://arxiv.org/abs/1409.1556.pdf)
- [VGG 19](http://arxiv.org/abs/1409.1556.pdf)


![](http://upload-images.jianshu.io/upload_images/145616-131a561dcbe74aba.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

(注：图片来自 [Github : TensorFlow-Slim image classification library](https://github.com/tensorflow/models/tree/master/slim#Pretrained))

额外参考：
- [[ILSVRC] 基于OverFeat的图像分类、定位、检测](http://www.jianshu.com/p/6d441e208547)
- [[卷积神经网络-进化史] 从LeNet到AlexNet](http://www.jianshu.com/p/7975f179ec49)
- [[透析] 卷积神经网络CNN究竟是怎样一步一步工作的？](http://www.jianshu.com/p/fe428f0b32c1)
- [GoogLenet中，1X1卷积核到底有什么作用呢？](http://www.jianshu.com/p/ba51f8c6e348)
- [深度学习 — 反向传播(BP)理论推导](http://www.jianshu.com/p/408ab8177a53)
- [无痛的机器学习第一季目录 - 知乎](https://zhuanlan.zhihu.com/p/22464594?refer=hsmyy)

### Object Detection

- [R-CNN](https://arxiv.org/abs/1311.2524)
- [Fast R-CNN](https://arxiv.org/abs/1504.08083)
- [Faster R-CNN](https://arxiv.org/abs/1506.01497v3)
- [FCN](https://arxiv.org/abs/1411.4038)
- [R-FCN](https://arxiv.org/abs/1605.06409v2)
- [YOLO](https://arxiv.org/abs/1506.02640v5)
- [SSD](https://arxiv.org/abs/1512.02325)

额外参考：
- [[Detection] CNN 之 "物体检测" 篇](http://www.jianshu.com/p/067f6a989d31)
- [计算机视觉中 RNN 应用于目标检测](http://www.jianshu.com/p/7e52daaba512)
- [Machine Learning 硬件投入调研](http://www.jianshu.com/p/4ce0aba4e3c2)

### RNN & LSTM

- [[福利] 深入理解 RNNs & LSTM 网络学习资料](http://www.jianshu.com/p/c930d61e1f16) @ [zhwhong](http://www.jianshu.com/u/38cd2a8c425e)
- [[RNN] Simple LSTM代码实现 & BPTT理论推导](http://www.jianshu.com/p/2aca6e8ac7c8) @ [zhwhong](http://www.jianshu.com/u/38cd2a8c425e)
- [计算机视觉中 RNN 应用于目标检测](http://www.jianshu.com/p/7e52daaba512) @ [zhwhong](http://www.jianshu.com/u/38cd2a8c425e)
- [推荐] [**Understanding LSTM Networks**](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) @ [colah](http://colah.github.io/) | [**理解LSTM网络**](http://www.jianshu.com/p/9dc9f41f0b29)[简书] @ [Not_GOD](http://www.jianshu.com/u/696dc6c6f01c)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) @ [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/)
- [LSTM Networks for Sentiment Analysis](http://deeplearning.net/tutorial/lstm.html) (theano官网LSTM教程+代码)
- [Recurrent Neural Networks Tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) @ [WILDML](http://www.wildml.com/)
- [Anyone Can Learn To Code an LSTM-RNN in Python (Part 1: RNN)](http://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/) @ [iamtrask](https://twitter.com/iamtrask)

## Stanford 机器学习课程整理

- [[coursera 机器学习课程] Machine Learning by Andrew Ng](http://www.jianshu.com/p/c68d0df13e0b) @ [zhwhong](http://www.jianshu.com/u/38cd2a8c425e)
- [[斯坦福CS231n课程整理] Convolutional Neural Networks for Visual Recognition（附翻译，下载）](http://www.jianshu.com/p/182baeb82c71) @ [zhwhong](http://www.jianshu.com/u/38cd2a8c425e)
- [[斯坦福CS224d课程整理] Natural Language Processing with Deep Learning](http://www.jianshu.com/p/062d2bbbef93) @ [zhwhong](http://www.jianshu.com/u/38cd2a8c425e)
- [[斯坦福CS229课程整理] Machine Learning Autumn 2016](http://www.jianshu.com/p/0a6ef31ff77a) @ [zhwhong](http://www.jianshu.com/u/38cd2a8c425e)

---

( 个人整理，未经允许禁止转载，授权转载请注明作者及出处，谢谢！)
