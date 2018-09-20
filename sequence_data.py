#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
 Filename @ sequence_data.py
 Author @ huangjunheng
 Create date @ 2018-02-26 15:19:27
 Description @ 
"""
import random
import tensorflow as tf
from tensorflow.contrib import rnn


def cal_model_para(filename):
    """
    根据数据计算模型的参数
    1. 最大sequence长度: max_seq_len
    2. 单个输入特征的维度: input_size
    3. label的维度，几分类就几个维度: num_class
    :param filename: 
    :return: 
    """
    max_seq_len = -1
    fr = open(filename)
    for i, line in enumerate(fr):
        line = line.rstrip('\n')
        data_split = line.split('&')
        feature_data_list = data_split[0].split('\t')

        if i == 0:
            input_size = len(feature_data_list[0].split('#'))
            num_class = len(data_split[1].split('\t'))

        cur_seq_len = len(feature_data_list)
        if cur_seq_len > max_seq_len:
            max_seq_len = cur_seq_len

    if max_seq_len % 10 != 0:
        max_seq_len = (int(max_seq_len / 10) + 1) * 10

    print('According to "%s", seq_max_len is set to %d, ' \
          'input_size is set to %d, num_class is set to %d.' \
          % (filename, max_seq_len, input_size, num_class))
    return max_seq_len, input_size, num_class


# ====================
#  Sequence Data
# ====================
class SequenceData(object):
    """ 
    判断序列是随机的还是有顺序的，序列的长度是不定的
    Generate or read sequence of data with dynamic length.
    For example:
    - Class 0: linear sequences (i.e. [0.1, 0.2, 0.3, 0.4,...])
    - Class 1: random sequences (i.e. [0.23, 0.3, 0.1, 0.87,...])

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """

    def __init__(self, filename, max_seq_len):
        self.batch_id = 0
        self.data, self.labels, self.seqlen = self.load_data(filename, max_seq_len)

    def next(self, batch_size):
        """ 
        获取全量数据(长度为n_samples)中的批量数据(长度为batch_size)
         e.g. n_samples = 100, batch_size = 16, batch_num = 7(6+1), last_batch_size = 4
        Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0

        batch_index = min(self.batch_id + batch_size, len(self.data))

        batch_data = (self.data[self.batch_id: batch_index])
        batch_labels = (self.labels[self.batch_id: batch_index])
        batch_seqlen = (self.seqlen[self.batch_id: batch_index])

        self.batch_id = batch_index
        return batch_data, batch_labels, batch_seqlen

    def cal_max_seq_len(self, filename):
        """
        计算最大sequence长度
        :param filename: 
        :return: 
        """
        max_seq_len = -1
        fr = open(filename)
        for line in fr:
            line = line.rstrip('\n')
            data_split = line.split('&')
            feature_data_list = data_split[0].split('\t')
            cur_seq_len = len(feature_data_list)
            if cur_seq_len > max_seq_len:
                max_seq_len = cur_seq_len

        if max_seq_len % 10 != 0:
            max_seq_len = ((max_seq_len / 10) + 1) * 10

        return max_seq_len

    def load_data(self, filename, max_seq_len=20):
        """
        加载数据
        :return: 
        """
        fr = open(filename)
        datas = []
        labels = []
        seqlen = []
        for line in fr:
            line = line.rstrip('\n')
            data_split = line.split('&')
            feature_data_list = data_split[0].split('\t')
            cur_seq_len = len(feature_data_list)
            seqlen.append(cur_seq_len)

            input_size = len(feature_data_list[0].split('#'))
            s = [[float(i) for i in item.split('#')] for item in feature_data_list]
            s += [[0.] * input_size for i in range(max_seq_len - cur_seq_len)]
            datas.append(s)

            if len(data_split) > 1: # 区分训练与预测
                label_data_list = data_split[1].split('\t')
                labels.append([float(item) for item in label_data_list])

        return datas, labels, seqlen

    def _data_generator(self, n_samples=500, max_seq_len=20, min_seq_len=3,
                        max_value=1000):
        """
        序列数据生成器
        :return: 
        """
        for i in range(n_samples):
            # Random sequence length
            len = random.randint(min_seq_len, max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(len)
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5:
                # Generate a linear sequence
                rand_start = random.randint(0, max_value - len)
                s = [[float(i) / max_value] for i in
                     range(rand_start, rand_start + len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                # Generate a random sequence
                s = [[float(random.randint(0, max_value)) / max_value]
                     for i in range(len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])

    def test(self):
        """
        测试
        :return: 
        """
        filename = 'data/test_data.txt'
        max_seq_len = self.cal_max_seq_len(filename)
        self.load_data(filename, max_seq_len)


if __name__ == '__main__':
    # s_data = SequenceData()
    # s_data.test()
    filename = 'data/test_data.txt'
    cal_model_para(filename=filename)
