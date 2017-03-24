---
title: GPU和CPU服务器测试mnist手写数字集
date: 2017-03-13 18:22:15
tags: [Deep Learning, GPU, TensorFlow]
categories: Machine Learning
---

# 一、GPU服务器

服务器 IP ：`172.xx.xx.98` （4块NVIDIA **TITAN X** GPU，**32** CPU核心）

<!--more-->

```bash
zhwhong@news-ai:~$ nvidia-smi
Mon Mar 13 14:30:39 2017       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 367.48                 Driver Version: 367.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX TIT...  Off  | 0000:01:00.0     Off |                  N/A |
| 22%   53C    P0    69W / 250W |      0MiB / 12206MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX TIT...  Off  | 0000:02:00.0     Off |                  N/A |
| 22%   57C    P0    72W / 250W |      0MiB / 12206MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX TIT...  Off  | 0000:82:00.0     Off |                  N/A |
| 22%   57C    P0    73W / 250W |      0MiB / 12206MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  GeForce GTX TIT...  Off  | 0000:83:00.0     Off |                  N/A |
|  0%   53C    P0    60W / 250W |      0MiB / 12206MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

```bash
zhwhong@news-ai:~$ lscpu
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                32
On-line CPU(s) list:   0-31
Thread(s) per core:    2
Core(s) per socket:    8
Socket(s):             2
NUMA node(s):          2
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 63
Stepping:              2
CPU MHz:               1201.218
BogoMIPS:              4800.94
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              20480K
NUMA node0 CPU(s):     0-7,16-23
NUMA node1 CPU(s):     8-15,24-31
```

使用 `cat /proc/cpuinfo`  命令可以查看每一个cpu核详细信息.

# 二、CPU服务器

服务器 IP ：`113.xx.xxx.196` （纯CPU服务器，**128核**）

```bash
mye@ubuntu:~$ lscpu
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                128
On-line CPU(s) list:   0-127
Thread(s) per core:    1
Core(s) per socket:    16
Socket(s):             8
NUMA node(s):          8
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 63
Stepping:              4
CPU MHz:               1200.031
BogoMIPS:              4396.82
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              40960K
NUMA node0 CPU(s):     0-15
NUMA node1 CPU(s):     16-31
NUMA node2 CPU(s):     32-47
NUMA node3 CPU(s):     48-63
NUMA node4 CPU(s):     64-79
NUMA node5 CPU(s):     80-95
NUMA node6 CPU(s):     96-111
NUMA node7 CPU(s):     112-127
```

使用 `cat /proc/cpuinfo`  命令可以查看每一个cpu核详细信息.

# 三、mnist测试

- 测试代码： [**zhwhong/awesome-deep-learning/TensorFlow-Tutorials**](https://github.com/Hzwcode/awesome-deep-learning/tree/master/TensorFlow-Tutorials)

## (1)逻辑回归logistic测试

Example: [**02_logistic_regression.py**](https://github.com/Hzwcode/awesome-deep-learning/blob/master/TensorFlow-Tutorials/02_logistic_regression.py)

测试结果：

### a.batch_size : 128

| --- | GPU | CPU |
|:---:|:---:|:---:|
| top信息 | %CPU：244.2 | %CPU：472 |
| nvidia-smi信息 | 20%左右 | 无 |
| mnist运行结果 | (99, 0.9234, </br> datetime.timedelta(0, 68, 913616)) </br> **统计：68s/100轮** | (99, 0.92330000000000001, </br> datetime.timedelta(0, 101, 424780)) </br> **统计：101s/100轮** |

### b.batch_size : 256

| --- | GPU | CPU |
|:---:|:---:|:---:|
| top信息 | %CPU：214.1 | %CPU：781.1 |
| nvidia-smi信息 | 24%左右 | 无 |
| mnist运行结果 | (99, 0.92290000000000005, datetime.timedelta(0, 45, 724627)) </br> **统计：45s/100轮** | (99, 0.92300000000000004, </br> datetime.timedelta(0, 79, 207202)) </br> **统计：79s/100轮** |

### c.batch_size : 512

| --- | GPU | CPU |
|:---:|:---:|:---:|
| top信息 | %CPU：203.2 | %CPU：1031 |
| nvidia-smi信息 | 29%左右 | 无 |
| mnist运行结果 | (99, 0.92000000000000004, datetime.timedelta(0, 30, 479467)) </br> **统计：30s/100轮** | (99, 0.92010000000000003,  </br> datetime.timedelta(0, 66, 738092)) </br> **统计：66秒/100轮** |

**GPU运行结果：**

![](logistic_gpu_1.png)

```bash
zhwhong@news-ai:~/MNIST_test$ nvidia-smi
Mon Mar 13 15:13:32 2017       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 367.48                 Driver Version: 367.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX TIT...  Off  | 0000:01:00.0     Off |                  N/A |
| 22%   57C    P2    70W / 250W |  11664MiB / 12206MiB |     29%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX TIT...  Off  | 0000:02:00.0     Off |                  N/A |
| 22%   58C    P2    71W / 250W |  11603MiB / 12206MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX TIT...  Off  | 0000:82:00.0     Off |                  N/A |
| 22%   57C    P2    71W / 250W |  11603MiB / 12206MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  GeForce GTX TIT...  Off  | 0000:83:00.0     Off |                  N/A |
| 22%   55C    P2    75W / 250W |  11601MiB / 12206MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0     28564    C   python                                       11660MiB |
|    1     28564    C   python                                       11599MiB |
|    2     28564    C   python                                       11599MiB |
|    3     28564    C   python                                       11597MiB |
+-----------------------------------------------------------------------------+
```

![](logistic_gpu_2.png)

**CPU运行结果：**

![](logistic_cpu_1.png)

![](logistic_cpu_2.png)

## (2)卷积神经网络conv测试

Example : [**05_convolutional_net.py**](https://github.com/Hzwcode/awesome-deep-learning/blob/master/TensorFlow-Tutorials/05_convolutional_net.py)

测试结果：

### a.batch_size : 128

| --- | GPU | CPU |
|:---:|:---|:---|
| top信息 | %CPU：141.9 | %CPU：5224.3 |
| nvidia-smi信息 | 75%左右 | 无 |
| mnist运行结果 |(0, 0.93359375, 4, 230888) </br> (1, 0.984375, 7, 929353) </br> (2, 0.97265625, 11, 635471) </br> (3, 0.98828125, 15, 310449) </br> (4, 0.9921875, 19, 3371) </br> (5, 0.98828125, 22, 720680) </br> (6, 1.0, 26, 384165) </br> (7, 0.99609375, 30, 88245) </br> …… </br> (99, 0.9921875, 370, 693523) </br> **平均：3.7s/轮** |(0, 0.95703125,  54, 907580) </br> (1, 0.98046875, 111, 935452) </br> (2, 0.98828125, 169, 417860) </br> (3, 0.98046875, 227, 60819) </br> (4, 0.9921875, 284, 513000) </br> (5, 0.98828125, 342, 273721) </br> (6, 0.9921875, 399, 981951) </br> (7, 0.984375, 458, 23667) </br> (8, 0.99609375, 516, 282659) </br> …… </br> **平均：57s/轮**|

### b.batch_size : 256

| --- | GPU | CPU |
|:---:|:---|:---|
| top信息 | %CPU：114.4 | %CPU：5746 |
| nvidia-smi信息 | 82%左右 | 无 |
| mnist运行结果 |(0, 0.6796875, 3, 563670) </br> (1, 0.9609375, 6, 565172) </br> (2, 0.96875, 9, 520787) </br> (3, 0.98828125, 12, 552352) </br> (4, 0.9921875, 15, 509898) </br> (5, 0.984375, 18, 508712) </br> (6, 0.9921875, 21, 465722) </br> …… </br> (99, 1.0, 301, 239776) </br> **平均：3s/轮**|(0, 0.69921875, 37, 712726) </br> (1, 0.97265625, 75, 387519) </br> (2, 0.984375, 113, 36748) </br> (3, 0.98828125, 150, 694555) </br> (4, 0.98828125, 188, 393595) </br> (5, 0.984375, 225, 962947) </br> (6, 0.98046875, 263, 551988) </br> (7, 0.9921875, 301, 107670) </br> …… </br> **平均：37s/轮**|

### c.batch_size : 512

| --- | GPU | CPU |
|:---:|:---|:---|
| top信息 | %CPU：98.5 | %CPU：5994 |
| nvidia-smi信息 | 90%左右 | 无 |
| mnist运行结果 | (0, 0.09375, 3, 358815) </br> (1, 0.52734375, 5, 918648) </br> (2, 0.91796875, 8, 488475) </br> (3, 0.9296875, 11, 35129) </br> (4, 0.98046875, 13, 605235) </br> (5, 0.96875, 16, 148614) </br> (6, 0.984375, 18, 715051) </br> (7, 0.9765625, 21, 281468) </br> (8, 0.9921875, 23, 854374) </br> …… </br> (99, 1.0, 263, 28433) </br> **平均：2.63s/轮**|(0, 0.08203125, 31, 125486) </br> (1, 0.796875, 62, 543181) </br> (2, 0.91015625, 94, 522874) </br> (3, 0.9609375, 126, 946088) </br> (4, 0.96484375, 159, 929706) </br> (5, 0.95703125, 193, 230872) </br> (6, 0.9921875, 226, 695604) </br> (7, 0.98828125, 260, 43828) </br> (8, 0.9921875, 293, 214191) </br> (9, 0.99609375, 326, 797200) </br> …… </br> **平均：32.6s/轮**|

**GPU运行结果：**

![](conv_gpu_1.png)

```bash
zhwhong@news-ai:~/MNIST_test$ nvidia-smi
Mon Mar 13 15:44:49 2017       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 367.48                 Driver Version: 367.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX TIT...  Off  | 0000:01:00.0     Off |                  N/A |
| 27%   70C    P2   192W / 250W |  11713MiB / 12206MiB |     90%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX TIT...  Off  | 0000:02:00.0     Off |                  N/A |
| 22%   53C    P2    70W / 250W |  11603MiB / 12206MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX TIT...  Off  | 0000:82:00.0     Off |                  N/A |
| 22%   45C    P2    69W / 250W |  11627MiB / 12206MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  GeForce GTX TIT...  Off  | 0000:83:00.0     Off |                  N/A |
| 22%   52C    P5    22W / 250W |  11601MiB / 12206MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0      9587    C   python                                       11709MiB |
|    1      9587    C   python                                       11599MiB |
|    2      1552    C   python                                         506MiB |
|    2      9587    C   python                                       11117MiB |
|    3      9587    C   python                                       11597MiB |
+-----------------------------------------------------------------------------+
```

![](conv_gpu_2.png)

**CPU运行结果：**

![](conv_cpu_1.png)

![](conv_cpu_2.png)

## (3)循环神经网络lstm测试

Example : [**07_lstm.py**](https://github.com/Hzwcode/awesome-deep-learning/blob/master/TensorFlow-Tutorials/07_lstm.py)

测试结果：

### batch_size : 512

| --- | GPU | CPU |
|:---:|:---|:---|
| top信息 | %CPU：123.4 | %CPU：818.4 |
| nvidia-smi信息 | 40%左右 | 无 |
| mnist运行结果 | (0, 0.26953125, 2, 390310) </br> (1, 0.37890625, 4, 420676) </br> (2, 0.68359375, 6, 385682) </br> (3, 0.7421875, 8, 494356) </br> (4, 0.7890625, 10, 649750) </br> (5, 0.84375, 12, 547186) </br> (6, 0.83203125, 14, 657817) </br> (7, 0.8671875, 16, 743615) </br> (8, 0.87109375, 18, 737803) </br> …… </br> …… </br> (99, 0.96875, 202, 633241) </br> **平均：2.02s/轮**|(0, 0.2265625, 10, 367446) </br> (1, 0.3984375, 20, 716101) </br> (2, 0.61328125, 31, 403893) </br> (3, 0.734375, 42, 7851) </br> (4, 0.75, 52, 698565) </br> (5, 0.78515625, 63, 61517) </br> (6, 0.84765625, 73, 529780) </br> (7, 0.84765625, 84, 130221) </br> (8, 0.8828125, 94, 898270) </br> (9, 0.90234375, 105, 455608) </br> …… </br> (99, 0.98046875, 995, 356187) </br> **平均：9.95s/轮**|

**GPU运行结果：**

![](lstm_gpu_1.png)

```bash
zhwhong@news-ai:~/MNIST_test$ nvidia-smi
Mon Mar 13 16:05:19 2017       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 367.48                 Driver Version: 367.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX TIT...  Off  | 0000:01:00.0     Off |                  N/A |
| 22%   61C    P2    90W / 250W |    185MiB / 12206MiB |     40%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX TIT...  Off  | 0000:02:00.0     Off |                  N/A |
| 22%   55C    P5    20W / 250W |    109MiB / 12206MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX TIT...  Off  | 0000:82:00.0     Off |                  N/A |
| 22%   55C    P5    56W / 250W |    109MiB / 12206MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  GeForce GTX TIT...  Off  | 0000:83:00.0     Off |                  N/A |
| 22%   54C    P5    21W / 250W |    109MiB / 12206MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0     17988    C   python                                         183MiB |
|    1     17988    C   python                                         107MiB |
|    2     17988    C   python                                         107MiB |
|    3     17988    C   python                                         107MiB |
+-----------------------------------------------------------------------------+
```

![](lstm_gpu_2.png)

**CPU运行结果：**

![](lstm_cpu_1.png)

![](lstm_cpu_2.png)

---

注：关于训练中每个epoch时间统计，可以使用python `datetime` 模块，使用`datetime.datetime.now()` 获取系统时间。
