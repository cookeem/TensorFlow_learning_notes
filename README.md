# tensorflow学习笔记

- 学习内容来源于：[caicloud / tensorflow-tutorial](https://github.com/caicloud/tensorflow-tutorial)

- 电子书：《Tensorflow实战Google深度学习框架》

![Tensorflow实战Google深度学习框架](images/cover.JPEG)

对样例中python2.x代码以及注释进行补充和修正，修正为python3.x代码。代码使用[jupyter/jupyter](https://github.com/jupyter/jupyter)进行交互演示，运行以下命令演示：

```sh
git clone https://github.com/cookeem/TensorFlow_learning_notes
cd TensorFlow_learning_notes
jupyter notebook
```

## 目录：
1. [第3章 TensorFlow入门](Chapter03)
1. [第4章 深层神经网络](Chapter04)
1. [第5章 MNIST数字识别问题](Chapter05)
1. [第6章 图像识别与卷积神经网络](Chapter06)
1. [第7章 图像数据处理](Chapter07)
1. [第8章 循环神经网络](Chapter08)
1. [第9章 TensorBoard可视化](Chapter09)
1. [第10章 TensorFlow计算加速](Chapter10)

## 学习笔记：

### 1、多层：使用多层权重，例如多层全连接方式
> 以下定义了三个隐藏层的全连接方式的神经网络
> 样例代码：

```python
import tensorflow as tf

l1 = tf.matmul(x, w1)
l2 = tf.matmul(l1, w2)
y = tf.matmul(l2,w3)
```

### 2、激活层：引入激活函数，让每一层去线性化
> 激活函数有多种，例如常用的：
> tf.nn.relu
> tf.nn.tanh
> tf.nn.sigmoid
> tf.nn.elu
> 样例代码：

```python
import tensorflow as tf

a = tf.nn.relu(tf.matmul(x, w1) + biase1)
y = tf.nn.relu(tf.matmul(a, w2) + biase2)
```

### 3、损失函数：
> 经典损失函数，交叉熵（cross entropy）
> 用于计算预测结果矩阵Y和实际结果矩阵Y_之间的距离
> 样例代码：

```python
import tensorflow as tf

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
```

```python
import tensorflow as tf

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tf.reduce_mean(tf.clip_by_value(v, 0.0, 10.0))
```

> 对于分类问题，通常把交叉熵与softmax回归一起使用

```python
import tensorflow as tf

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
```

> 对于回归问题，通常使用mse（均方误差函数）计算损失函数

```python
import tensorflow as tf

mse_loss = tf.reduce_mean(tf.square(y_ - y))

# 与以下函数计算结果完全一致
dataset_size = 1000
mse_loss = tf.reduce_sum(tf.pow(y_ - y, 2)) / dataset_size
```

> 自定义条件化的损失函数

```python
import tensorflow as tf

loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
```

### 4、神经网络优化算法，训练优化器
> 一般优化器的目标是优化权重W和偏差biases，最小化损失函数的结果
> 以下优化器会不断优化W和biases

```python
import tensorflow as tf

LEARNING_RATE = 0.001
mse_loss = tf.reduce_mean(tf.square(y_ - y))
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(mse_loss)
```

### 5、优化学习率LEARNING_RATE
> 学习率设置过大可能导致无法收敛，学习率设置过小可能导致收敛过慢

```python
import tensorflow as tf

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
    learning_rate=0.1, 
    global_step=global_step, 
    decay_steps=100, 
    decay_rate=0.96, 
    staircase=True, 
    name=None
)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
```

### 6、过拟合问题（正则化）
> 避免训练出来的模型过分复杂，即模型记住了所有数据（包括噪声引起的误差）
> 因此需要引入正则化函数叠加的方式，避免模型出现过拟合

```python
import tensorflow as tf

v_lambda = 0.001
w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w)
mse_loss = tf.reduce_mean(tf.square(y_ - y) + tf.contrib.layers.l2_regularizer(v_lambda)(w))
```

### 7、滑动平均模型
> 用于控制模型的变化速度，可以控制权重W以及偏差biases
> 例如：avg_class.average(w) avg_class.average(biases)

```python
import tensorflow as tf

v1 = tf.Variable(0, dtype=tf.float32)
step = tf.Variable(0, trainable=False)
ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=step)
# 每一次操作的时候，列表变量[v1]都会被更新
maintain_averages_op = ema.apply([v1]) 

with tf.Session() as sess:
    
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run([v1, ema.average(v1)]))
    
    # 更新step和v1的取值
    sess.run(tf.assign(step, 10000))  
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
```
