# Dynamic-RNN-LSTM-GRU
基于tensorflow的动态双向rnn, lstm及gru的实现

Implementation of dynamic bi_directional rnn, lstm and gru based on tensorflow


## 当前功能
- [x] 支持特征序列长度不一致
- [x] 支持rnn, lstm, gru三种模型配置
- [x] 根据训练数据动态适配模型参数
- [x] 支持双向rnn等
- [x] 支持dropout层


## 运行
```
python dynamic_sequence_model.py
```

## 运行结果
```
Step 1, Minibatch Loss= 0.738726, Training Accuracy= 0.44531
Step 200, Minibatch Loss= 0.696200, Training Accuracy= 0.53846
... ... ...
Step 3600, Minibatch Loss= 0.326143, Training Accuracy= 0.88462
Step 3800, Minibatch Loss= 0.271917, Training Accuracy= 0.91346
Step 4000, Minibatch Loss= 0.230929, Training Accuracy= 0.93269
Optimization Finished!
Testing Accuracy: 0.926
Predict Result: [[0, 1], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 1]]
```

## 参数设置
1. 模型参数文件：*model_config.py*；
2. 一些参数如： **seq_max_len**, **input_size**, **num_class** 需要根据数据动态算出；

## Demo数据介绍
判断序列是随机的还是有顺序的，序列的长度是不定的。
```
For example:

Class 0: linear sequences (i.e. [0.1, 0.2, 0.3, 0.4,...])

Class 1: random sequences (i.e. [0.23, 0.3, 0.1, 0.87,...])
```
## 数据输入格式
1. **training data and test data**
```
训练数据及测试数据的输入格式由两部分组成: Features_line + '&' + Labels_line

1. Features_line:
   feature_num (feature_num可以不一样）个feature，用 '\t' 隔开.
   其中feature可以有input_size(input_size需一致)个维度，每个维度用 '#' 隔开.
2. Labels_line:
   label_num(label_num需一致)个label(one-hot形式)，用 '\t' 隔开.

例如：

当input_size=1，有：
1   2   3&1 0
2   7   4   8&0 1

当input_size=2，有：
1#3   2#5   3#7&1 0
2#1   7#45   4#89   8#92&0 1
```
2. **predict data**
```
预测数据输入格式只由一部分组成，Features_line

1. Features_line:
   feature_num (feature_num可以不一样）个feature，用 '\t' 隔开.
   其中feature可以有input_size(input_size需一致)个维度，每个维度用 '#' 隔开.


例如：

当input_size=1，有：
1   2   3
2   7   4   8

当input_size=2，有：
1#3   2#5   3#7
2#1   7#45   4#89   8#92
```

## 参考
1. https://github.com/aymericdamien/TensorFlow-Examples/



