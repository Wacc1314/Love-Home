Python-ELM v0.3
===============

######This is an implementation of the [Extreme Learning Machine](http://www.extreme-learning-machines.org) [1][2] in Python, based on [scikit-learn](http://scikit-learn.org).

######From the abstract:

> It is clear that the learning speed of feedforward neural networks is in general far slower than required and it has been a major bottleneck in their applications for past decades. Two key reasons behind may be: 1) the slow gradient- based learning algorithms are extensively used to train neural networks, and 2) all the parameters of the networks are tuned iteratively by using such learning algorithms. Unlike these traditional implementations, this paper proposes a new learning algorithm called extreme learning machine (ELM) for single- hidden layer feedforward neural networks (SLFNs) which ran- domly chooses the input weights and analytically determines the output weights of SLFNs. In theory, this algorithm tends to provide the best generalization performance at extremely fast learning speed. The experimental results based on real- world benchmarking function approximation and classification problems including large complex applications show that the new algorithm can produce best generalization performance in some cases and can learn much faster than traditional popular learning algorithms for feedforward neural networks.
>>很明显，前馈神经网络的学习速度通常远远慢于所需要的速度，并且它已经是它们在过去几十年的应用中的主要瓶颈。背后的两个关键原因可能是：1）基于慢速梯​​度的学习算法广泛地用于训练神经网络，以及2）通过使用这种学习算法迭代地调谐网络的所有参数。与这些传统的实现不同，本文提出了一种新的学习算法，称为极限学习机（ELM）用于单隐层前馈神经网络（SLFN），其随机选择输入权重并分析确定SLFN的输出权重。理论上，该算法倾向于在极快的学习速度下提供最佳的泛化性能。基于真实世界基准函数近似和包括大型复杂应用的分类问题的实验结果表明，新算法在某些情况下可以产生最佳的泛化性能，并且比前馈神经网络的传统流行学习算法学习得快得多。

It's a work in progress, so things can/might/will change.

__David C. Lambert__  
__dcl [at] panix [dot] com__  

__Copyright © 2013__  
__License: Simple BSD__

Files
-----
####__random_layer.py__

Contains the __RandomLayer__, __MLPRandomLayer__, __RBFRandomLayer__ and __GRBFRandomLayer__ classes.
>包含RandomLayer，MLPRandomLayer，RBFRandomLayer和GRBFRandomLayer类。

RandomLayer is a transformer that creates a feature mapping of the inputs that corresponds to a layer of hidden units with randomly  generated components.
>RandomLayer是一个变换器，它使用随机生成的组件创建对应于一层隐藏单元的输入的要素映射。

The transformed values are a specified function of input activations that are a weighted combination of dot product (multilayer perceptron) and distance (rbf) activations:
>变换值是输入激活的指定函数，其是点积（多层感知器）和距离（rbf）激活的加权组合：

	  input_activation = alpha * mlp_activation + (1-alpha) * rbf_activation

	  mlp_activation(x) = dot(x, weights) + bias
	  rbf_activation(x) = rbf_width * ||x - center||/radius

_mlp_activation_ is multi-layer perceptron input activation  
>mlp_activation是多层感知器输入激活

_rbf_activation_ is radial basis function input activation
>rbf_activation是径向基函数输入激活

_alpha_ and _rbf_width_ are specified by the user
>alpha和rbf_width由用户指定

_weights_ and _biases_ are taken from normal distribution of mean 0 and sd of 1
>权重和偏差取自平均值0和sd为1的正态分布

_centers_ are taken uniformly from the bounding hyperrectangle of the inputs, and
>中心从输入的有限超矩形均匀地取出，

	radius = max(||x-c||)/sqrt(n_centers*2)

(All random components can be supplied by the user by providing entries in the dictionary given as the _user_components_ parameter.)

The input activation is transformed by a transfer function that defaults
to numpy.tanh if not specified, but can be any callable that returns an
array of the same shape as its argument (the input activation array, of
shape [n_samples, n_hidden]).
>输入激活通过传递函数转换，默认为numpy.tanh（如果未指定），但可以是返回与其参数（输入激活数组，形状[n_samples，n_hidden]）相同形状的数组的任何可调用函数。

Transfer functions provided are:

*	sine
*	tanh
*	tribas
*	inv_tribas
*	sigmoid
*	hardlim
*	softlim
*	gaussian
*	multiquadric
*	inv_multiquadric

MLPRandomLayer and RBFRandomLayer classes are just wrappers around the RandomLayer class, with the _alpha_ mixing parameter set to 1.0 and 0.0 respectively (for 100% MLP input activation, or 100% RBF input activation)
>MLPRandomLayer和RBFRandomLayer类只是在RandomLayer类周围的包装器，其中_alpha_混合参数分别设置为1.0和0.0（对于100％MLP输入激活或100％RBF输入激活）

The RandomLayer, MLPRandomLayer, RBFRandomLayer classes can take a callable user
provided transfer function.  See the docstrings and the example ipython
notebook for details.
>RandomLayer，MLPRandomLayer，RBFRandomLayer类可以采用可调用的用户提供的传递函数。 有关详细信息，请参阅docstrings和示例ipython notebook。

The GRBFRandomLayer implements the Generalized Radial Basis Function from [[3]](http://sci2s.ugr.es/keel/pdf/keel/articulo/2011-Neurocomputing1.pdf)

####__elm.py__

Contains the __ELMRegressor__, __ELMClassifier__, __GenELMRegressor__, and __GenELMClassifier__ classes.

GenELMRegressor and GenELMClassifier both take *RandomLayer instances as part of their contructors, and an optional regressor (conforming to the sklearn API)for performing the fit (instead of the default linear fit using the pseudo inverse from scipy.pinv2).
>GenELMRegressor和GenELMClassifier都将* RandomLayer实例作为其构造器的一部分，以及用于执行拟合的可选回归（符合sklearn API）（而不是使用来自scipy.pinv2的伪逆的默认线性拟合）。

GenELMClassifier is little more than a wrapper around GenELMRegressor that binarizes the target array before performing a regression, then unbinarizes the prediction of the regressor to make its own predictions.
>GenELMClassifier比在执行回归之前对目标数组进行二进制化的GenELMRegressor包装器稍微多一些，然后对回归器的预测进行二值化以进行自己的预测。

The ELMRegressor class is a wrapper around GenELMRegressor that uses a RandomLayer instance by default and exposes the RandomLayer parameters in the constructor.  ELMClassifier is similar for classification.
>ELMRegressor类是GenELMRegressor的一个包装器，默认使用RandomLayer实例，并在构造函数中公开RandomLayer参数。 ELMClassifier类似于分类。

####__plot_elm_comparison.py__

A small demo (based on scikit-learn's plot_classifier_comparison) that shows the decision functions of a couple of different instantiations of the GenELMClassifier on three different datasets.
>一个小演示（基于scikit-learn的plot_classifier_comparison），显示了三个不同数据集上GenELMClassifier的几个不同实例化的决策函数。

####__elm_notebook.py__

An IPython notebook, illustrating several ways to use the __\*ELM\*__ and __\*RandomLayer__ classes.
>一个IPython笔记本，说明了使用__ \ * ELM \ * __和__ \ * RandomLayer__类的几种方法。

Requirements
------------

Written using Python 2.7.3, numpy 1.6.1, scipy 0.10.1, scikit-learn 0.13.1 and ipython 0.12.1

References
----------
```
[1] http://www.extreme-learning-machines.org

[2] G.-B. Huang, Q.-Y. Zhu and C.-K. Siew, "Extreme Learning Machine:
          Theory and Applications", Neurocomputing, vol. 70, pp. 489-501,
          2006.
          
[3] Fernandez-Navarro, et al, "MELM-GRBF: a modified version of the  
          extreme learning machine for generalized radial basis function  
          neural networks", Neurocomputing 74 (2011), 2502-2510
```

