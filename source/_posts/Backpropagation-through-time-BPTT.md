---
title: '[RNN] Simple LSTM代码实现 & BPTT理论推导'
date: 2017-02-24 18:26:58
tags: [RNN, LSTM, Algorithm]
categories: Machine Learning
mathjax: true
---

- 参考：[Nico's Blog - Simple LSTM](http://nicodjimenez.github.io/2014/08/08/lstm.html)
- Github代码：[https://github.com/Hzwcode/lstm](https://github.com/Hzwcode/lstm)

---

前面我们介绍过CNN中普通的[BP反向传播算法的推导](https://Hzwcode.github.io/2017/02/24/Backpropagation-principle/)，但是在RNN（比如[LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)）中，反向传播被称作[BPTT](https://en.wikipedia.org/wiki/Backpropagation_through_time)（Back Propagation Through Time），它是和时间序列有关的。

<!--more-->

![Back Propagation Through Time](http://upload-images.jianshu.io/upload_images/145616-113aeedc747a3628.gif?imageMogr2/auto-orient/strip)

![](http://upload-images.jianshu.io/upload_images/145616-4ae3ab8b8426cdcd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


A few weeks ago I released some [code](https://github.com/nicodjimenez/lstm) on Github to help people understand how LSTM’s work at the implementation level. The forward pass is well explained elsewhere and is straightforward to understand, but I derived the backprop equations myself and the backprop code came without any explanation whatsoever. The goal of this post is to explain the so called *backpropagation through time* in the context of LSTM’s.

If you feel like anything is confusing, please post a comment below or submit an issue on Github.

**Note:** this post assumes you understand the forward pass of an LSTM network, as this part is relatively simple. Please read this [great intro paper](http://arxiv.org/abs/1506.00019) if you are not familiar with this, as it contains a very nice intro to LSTM’s. I follow the same notation as this paper so I recommend reading having the tutorial open in a separate browser tab for easy reference while reading this post.

> # Introduction (Simple LSTM)

![LSTM Block](http://upload-images.jianshu.io/upload_images/145616-4951e5c5352a88f2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

The forward pass of an LSTM node is defined as follows:

![][01]
[01]:http://latex.codecogs.com/png.latex?\\g(t)%20&=&%20\phi(W_{gx}%20x(t)%20+%20W_{gh}%20h(t-1)%20+%20b_{g})%20\\\\%20i(t)%20&=&%20\sigma(W_{ix}%20x(t)%20+%20W_{ih}%20h(t-1)%20+%20b_{i})%20\\\\%20f(t)%20&=&%20\sigma(W_{fx}%20x(t)%20+%20W_{fh}%20h(t-1)%20+%20b_{f})%20\\\\%20o(t)%20&=&%20\sigma(W_{ox}%20x(t)%20+%20W_{oh}%20h(t-1)%20+%20b_{o})%20\\\\%20s(t)%20&=&%20g(t)%20*%20i(t)%20+%20s(t-1)%20*%20f(t)%20\\\\%20h(t)%20&=&%20s(t)%20*%20o(t)%20\\

(**注**：这里最后一个式子`h(t)`的计算，普遍认为`s(t)`前面还有一个tanh激活，然后再乘以`o(t)`，不过 peephole LSTM paper中建议此处激活函数采用 `f(x) = x`，所以这里就没有用`tanh`（下同），可以参见[Wiki - Long_short-term_memory](https://en.wikipedia.org/wiki/Long_short-term_memory)上面所说的)

![](http://upload-images.jianshu.io/upload_images/145616-ad0508a2df64e3ec.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

By concatenating the `x(t)` and `h(t-1)` vectors as follows:

![][02]
[02]:http://latex.codecogs.com/png.latex?x_c(t)%20=%20[x(t),%20h(t-1)]

we can rewrite parts of the above as follows:

![][03]
[03]:http://latex.codecogs.com/png.latex?\\g(t)%20&=&%20\phi(W_{g}%20x_c(t)%20+%20b_{g})%20\\\\%20i(t)%20&=&%20\sigma(W_{i}%20x_c(t)%20+%20b_{i})%20\\\\%20f(t)%20&=&%20\sigma(W_{f}%20x_c(t)%20+%20b_{f})%20\\\\%20o(t)%20&=&%20\sigma(W_{o}%20x_c(t)%20+%20b_{o})

Suppose we have a loss `l(t)` that we wish to minimize at every time step `t` that depends on the hidden layer `h` and the label `y` at the current time via a loss function `f`:

![][04]
[04]:http://latex.codecogs.com/png.latex?l(t)%20=%20f(h(t),%20y(t))

where `f` can be any differentiable loss function, such as the Euclidean loss:

![][05]
[05]:http://latex.codecogs.com/png.latex?l(t)%20=%20f(h(t),%20y(t))%20=%20\|%20h(t)%20-%20y(t)%20\|^2

Our ultimate goal in this case is to use gradient descent to minimize the loss `L` over an entire sequence of length `T`：

![][06]
[06]:http://latex.codecogs.com/png.latex?L%20=%20\sum_{t=1}^{T}%20l(t)

Let’s work through the algebra of computing the loss gradient:

![][07]
[07]:http://latex.codecogs.com/png.latex?\frac{dL}{dw}

where `w` is a scalar parameter of the model (for example it may be an entry in the matrix `W_gx`). Since the loss `l(t) = f(h(t),y(t))` only depends on the values of the hidden layer `h(t)` and the label `y(t)`, we have by the chain rule:

![][08]
[08]:http://latex.codecogs.com/png.latex?\frac{dL}{dw}%20=%20\sum_{t%20=%201}^{T}%20\sum_{i%20=%201}^{M}%20\frac{dL}{dh_i(t)}\frac{dh_i(t)}{dw}


where `h_i(t)` is the scalar corresponding to the `i’th` memory cell’s hidden output and `M` is the total number of memory cells. Since the network propagates information forwards in time, changing `h_i(t)` will have no effect on the loss prior to time `t`, which allows us to write:

![][09]
[09]:http://latex.codecogs.com/png.latex?\frac{dL}{dh_i(t)}%20=%20\sum_{s=1}^T%20\frac{dl(s)}{dh_i(t)}%20=%20\sum_{s=t}^T%20\frac{dl(s)}{dh_i(t)}

For notational convenience we introduce the variable `L(t)` that represents the cumulative loss from step tonwards:

![][10]
[10]:http://latex.codecogs.com/png.latex?L(t)%20=%20\sum_{s=t}^{s=T}%20l(s)

such that `L(1)` is the loss for the entire sequence. This allows us to rewrite the above equation as:

![][11]
[11]:http://latex.codecogs.com/png.latex?\frac{dL}{dh_i(t)}%20=%20\sum_{s=t}^T%20\frac{dl(s)}{dh_i(t)}%20=%20\frac{dL(t)}{dh_i(t)}

With this in mind, we can rewrite our gradient calculation as:

![][12]
[12]:http://latex.codecogs.com/png.latex?\frac{dL}{dw}%20=%20\sum_{t%20=%201}^{T}%20\sum_{i%20=%201}^{M}%20\frac{dL(t)}{dh_i(t)}\frac{dh_i(t)}{dw}

Make sure you understand this last equation. The computation of `dh_i(t) / dw` follows directly follows from the forward propagation equations presented earlier. We now show how to compute `dL(t) / dh_i(t)` which is where the so called ***backpropagation through time*** comes into play.

> # Backpropagation through time (BPTT)

![Back Propagation Through Time](http://upload-images.jianshu.io/upload_images/145616-113aeedc747a3628.gif?imageMogr2/auto-orient/strip)

This variable `L(t)` allows us to express the following recursion:

![][13]
[13]:http://latex.codecogs.com/png.latex?L(t)%20=%20\begin{cases}%20l(t)%20+%20L(t+1)%20&%20\text{if}%20\,%20t%20%3C%20T%20\\%20l(t)%20&%20\text{if}%20\,%20t%20=%20T%20\end{cases}

Hence, given activation `h(t)` of an LSTM node at time `t`, we have that:

![][14]
[14]:http://latex.codecogs.com/png.latex?\frac{dL(t)}{dh(t)}%20=%20\frac{dl(t)}{dh(t)}%20+%20\frac{dL(t+1)}{dh(t)}

Now, we know where the first term on the right hand side `dl(t) / dh(t)` comes from: it’s simply the elementwise derivative of the loss `l(t)` with respect to the activations `h(t)` at time `t`. The second term `dL(t+1) / dh(t)` is where the recurrent nature of LSTM’s shows up. It shows that the we need the *next* node’s derivative information in order to compute the current *current* node’s derivative information. Since we will ultimately need to compute `dL(t) / dh(t)` for all `t = 1, 2, ... , T`, we start by computing

![][15]
[15]:http://latex.codecogs.com/png.latex?\frac{dL(T)}{dh(T)}%20=%20\frac{dl(T)}{dh(T)}

and work our way backwards through the network. Hence the term *backpropagation through time*. With these intuitions in place, we jump into the code.

> # Code (Talk is cheap, Show me the code)

We now present the code that performs the backprop pass through a single node at time `1 <= t <= T`. The code takes as input:

![](http://upload-images.jianshu.io/upload_images/145616-98208ee1ecaa495f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

And computes:

![](http://upload-images.jianshu.io/upload_images/145616-28f4a30188b7dd3e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

whose values will need to be propagated backwards in time. The code also adds derivatives to:

![](http://upload-images.jianshu.io/upload_images/145616-8c47979fe8d86be1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

since recall that we must sum the derivatives from each time step:

![][16]
[16]:http://latex.codecogs.com/png.latex?\frac{dL}{dw}%20=%20\sum_{t%20=%201}^{T}%20\sum_{i%20=%201}^{M}%20\frac{dL(t)}{dh_i(t)}\frac{dh_i(t)}{dw}

Also, note that we use:

![](http://upload-images.jianshu.io/upload_images/145616-34424089b87efa6c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

where we recall that `X_c(t) = [x(t), h(t-1)]`. Without any further due, the code:

```python
def top_diff_is(self, top_diff_h, top_diff_s):
    # notice that top_diff_s is carried along the constant error carousel
    ds = self.state.o * top_diff_h + top_diff_s
    do = self.state.s * top_diff_h
    di = self.state.g * ds
    dg = self.state.i * ds
    df = self.s_prev * ds

    # diffs w.r.t. vector inside sigma / tanh function
    di_input = (1. - self.state.i) * self.state.i * di
    df_input = (1. - self.state.f) * self.state.f * df
    do_input = (1. - self.state.o) * self.state.o * do
    dg_input = (1. - self.state.g ** 2) * dg

    # diffs w.r.t. inputs
    self.param.wi_diff += np.outer(di_input, self.xc)
    self.param.wf_diff += np.outer(df_input, self.xc)
    self.param.wo_diff += np.outer(do_input, self.xc)
    self.param.wg_diff += np.outer(dg_input, self.xc)
    self.param.bi_diff += di_input
    self.param.bf_diff += df_input
    self.param.bo_diff += do_input
    self.param.bg_diff += dg_input

    # compute bottom diff
    dxc = np.zeros_like(self.xc)
    dxc += np.dot(self.param.wi.T, di_input)
    dxc += np.dot(self.param.wf.T, df_input)
    dxc += np.dot(self.param.wo.T, do_input)
    dxc += np.dot(self.param.wg.T, dg_input)

    # save bottom diffs
    self.state.bottom_diff_s = ds * self.state.f
    self.state.bottom_diff_x = dxc[:self.param.x_dim]
    self.state.bottom_diff_h = dxc[self.param.x_dim:]
```

> # Details

The forward propagation equations show that modifying `s(t)` affects the loss `L(t)` by directly changing the values of `h(t)` as well as `h(t+1)`. However, modifying `s(t)` affects `L(t+1)` only by modifying `h(t+1)`. Therefore, by the chain rule:

![][17]
[17]:http://latex.codecogs.com/png.latex?\\\frac{dL(t)}{ds_i(t)}%20=%20\frac{dL(t)}{dh_i(t)}%20\frac{dh_i(t)}{ds_i(t)}%20+%20\frac{dL(t)}{dh_i(t+1)}%20\frac{dh_i(t+1)}{ds_i(t)}%20\\\\\\=%20\frac{dL(t)}{dh_i(t)}%20\frac{dh_i(t)}{ds_i(t)}%20+%20\frac{dL(t+1)}{dh_i(t+1)}%20\frac{dh_i(t+1)}{ds_i(t)}%20\\\\\\=%20\frac{dL(t)}{dh_i(t)}%20\frac{dh_i(t)}{ds_i(t)}%20+%20\frac{dL(t+1)}{ds_i(t)}%20\\\\\\%20=%20\frac{dL(t)}{dh_i(t)}%20\frac{dh_i(t)}{ds_i(t)}%20+%20[\texttt{top\_diff\_s}]_i%20\\

Since the forward propagation equations state:

![][18]
[18]:http://latex.codecogs.com/png.latex?h(t)%20=%20s(t)%20*%20o(t)

we get that:

![][19]
[19]:http://latex.codecogs.com/png.latex?\frac{dL(t)}{dh_i(t)}%20*%20\frac{dh_i(t)}{ds_i(t)}%20=%20o_i(t)%20*%20[\texttt{top\_diff\_h}]_i

Putting all this together we have:

```python
ds = self.state.o * top_diff_h + top_diff_s
```

The rest of the equations should be straightforward to derive, please let me know if anything is unclear.

---



> # Test  LSTM Network

此 [代码](https://github.com/Hzwcode/lstm) 其是通过自己实现 lstm 网络来逼近一个序列，y_list = [-0.5, 0.2, 0.1, -0.5]，测试结果如下：

```
cur iter:  0
y_pred[0] : 0.041349
y_pred[1] : 0.069304
y_pred[2] : 0.116993
y_pred[3] : 0.165624
loss:  0.753483886253
cur iter:  1
y_pred[0] : -0.223297
y_pred[1] : -0.323066
y_pred[2] : -0.394514
y_pred[3] : -0.433984
loss:  0.599065083953
cur iter:  2
y_pred[0] : -0.140715
y_pred[1] : -0.181836
y_pred[2] : -0.219436
y_pred[3] : -0.238904
loss:  0.445095565699
cur iter:  3
y_pred[0] : -0.138010
y_pred[1] : -0.166091
y_pred[2] : -0.203394
y_pred[3] : -0.233627
loss:  0.428061605701
cur iter:  4
y_pred[0] : -0.139986
y_pred[1] : -0.157368
y_pred[2] : -0.195655
y_pred[3] : -0.237612
loss:  0.413581711096
cur iter:  5
y_pred[0] : -0.144410
y_pred[1] : -0.151859
y_pred[2] : -0.191676
y_pred[3] : -0.246137
loss:  0.399770442382
cur iter:  6
y_pred[0] : -0.150306
y_pred[1] : -0.147921
y_pred[2] : -0.189501
y_pred[3] : -0.257119
loss:  0.386136380384
cur iter:  7
y_pred[0] : -0.157119
y_pred[1] : -0.144659
y_pred[2] : -0.188067
y_pred[3] : -0.269322
loss:  0.372552465753
cur iter:  8
y_pred[0] : -0.164490
y_pred[1] : -0.141537
y_pred[2] : -0.186737
y_pred[3] : -0.281914
loss:  0.358993892096
cur iter:  9
y_pred[0] : -0.172187
y_pred[1] : -0.138216
y_pred[2] : -0.185125
y_pred[3] : -0.294326
loss:  0.345449256686
cur iter:  10
y_pred[0] : -0.180071
y_pred[1] : -0.134484
y_pred[2] : -0.183013
y_pred[3] : -0.306198
loss:  0.331888922037

……

cur iter:  97
y_pred[0] : -0.500351
y_pred[1] : 0.201185
y_pred[2] : 0.099026
y_pred[3] : -0.499154
loss:  3.1926009167e-06
cur iter:  98
y_pred[0] : -0.500342
y_pred[1] : 0.201122
y_pred[2] : 0.099075
y_pred[3] : -0.499190
loss:  2.88684626031e-06
cur iter:  99
y_pred[0] : -0.500331
y_pred[1] : 0.201063
y_pred[2] : 0.099122
y_pred[3] : -0.499226
loss:  2.61076360677e-06

```

可以看出迭代100轮，最后Loss在不断收敛，并且逐渐逼近了预期序列：y_list = [-0.5, 0.2, 0.1, -0.5]。

# Reference

- [深度学习 — 反向传播(BP)理论推导 (zhwhong)](http://www.jianshu.com/p/408ab8177a53)
- [Nico's Blog：Simple LSTM](http://nicodjimenez.github.io/2014/08/08/lstm.html)
- [Github仓库：https://github.com/Hzwcode/lstm](https://github.com/Hzwcode/lstm)
- [RECURRENT NEURAL NETWORKS TUTORIAL, PART 3 – BACKPROPAGATION THROUGH TIME AND VANISHING GRADIENTS](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)
- [[福利] 深入理解 RNNs & LSTM 网络学习资料](http://www.jianshu.com/p/c930d61e1f16)
- [关于简书中如何编辑Latex数学公式](http://www.jianshu.com/p/c7e3f417641c)

---

(转载请联系作者并注明出处，谢谢！)
