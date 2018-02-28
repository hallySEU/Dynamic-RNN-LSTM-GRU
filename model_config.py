#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
 Filename @ model_config.py
 Author @ huangjunheng
 Create date @ 2018-02-26 15:09:27
 Description @ config
"""

import tensorflow as tf
from tensorflow.contrib import rnn


class ModelConfig:
    """
    模型配置
    """
    # 定义训练参数
    learning_rate = 0.01
    training_steps = 4000
    display_steps = 200
    batch_size = 128

    # 定义隐藏层数目
    num_hidden = 64

    # Get from data dynamically
    # input_size = 1
    # seq_max_len = 20  # Sequence max length, 相对于static中的time_steps
    # num_class = 2

    # 使用何种模型
    model = 'gru'
    bi_directional = True

    # 数据位置
    training_data = 'data/training_data.txt'
    test_data = 'data/test_data.txt'
    predict_data = 'data/predict_data.txt'
