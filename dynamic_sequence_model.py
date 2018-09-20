#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
 Filename @ dynamic_sequence_model.py
 Author @ huangjunheng
 Create date @ 2018-02-26 15:09:27
 Description @ rnn, lstm, gru integrate
"""

from __future__ import print_function

# gpu使用
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1,2'

import tensorflow as tf
from tensorflow.contrib import rnn

import indicator

from model_config import ModelConfig
from sequence_data import SequenceData
from sequence_data import cal_model_para


class DynamicSequenceModel:
    """
    整合 dynamic (rnn, lstm, gru) 功能
    """
    def __init__(self):
        self.conf = ModelConfig()
        self.seq_max_len, self.input_size, self.num_class = cal_model_para(filename=self.conf.training_data)
        self._init_varible()
        self.loss_op, self.optimizer_op, self.accuracy_op, self.predict_op, \
            self.predict_pro_op = self.define_operator()

    def _init_varible(self):

        self.X = tf.placeholder(tf.float32, [None, self.seq_max_len, self.input_size])
        self.Y = tf.placeholder(tf.float32, [None, self.num_class])

        self.dropout_keep_prob = tf.placeholder(tf.float32)
        # A placeholder for indicating each sequence length
        self.seqlen = tf.placeholder(tf.int32, [None])

        self.biases = {
            'out': tf.Variable(tf.random_normal([self.num_class]))
        }
        self.weights = {
            'out': tf.Variable(tf.random_normal([self.conf.num_hidden, self.num_class]))
        }

        # 针对双向cell
        self.bi_weights = {
            'out': tf.Variable(tf.random_normal([self.conf.num_hidden * 2, self.num_class]))
        }

    def cell_operation(self, X):
        # 动态rnn提供了每个序列的长度信息，因为每个序列的长度不一样.
        # 每一个sample的output取这个sample的最后长度的cell的输出作为隐层输出，即output[cur_len-1]，而不是静态rnn的output[-1]
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(X, self.seq_max_len, axis=1)

        # Define a cell with tensorflow
        if self.conf.model == 'rnn':
            cell_func = rnn.BasicRNNCell
        elif self.conf.model == 'lstm':
            cell_func = rnn.BasicLSTMCell
        elif self.conf.model == 'gru':
            cell_func = rnn.GRUCell
        else:
            raise Exception("model type not supported: {}".format(self.conf.model))

        print('Use %s model, use bi_directional cell: %s.' % (self.conf.model, self.conf.use_bi_directional))

        if self.conf.use_bi_directional:
            # Forward direction cell
            fw_cell = cell_func(self.conf.num_hidden)
            fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            # Backward direction cell
            bw_cell = cell_func(self.conf.num_hidden)
            bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
            # Get bi-cell output
            try:
                outputs, _, _ = rnn.static_bidirectional_rnn(fw_cell, bw_cell, x,
                                                             dtype=tf.float32,
                                                             sequence_length=self.seqlen)
            except Exception:  # Old TensorFlow version only returns outputs not states
                outputs = rnn.static_bidirectional_rnn(fw_cell, bw_cell, x,
                                                       dtype=tf.float32,
                                                       sequence_length=self.seqlen)
        else:
            # Define a cell with tensorflow
            # Get cell output, providing 'sequence_length' will perform dynamic calculation.

            cell = cell_func(self.conf.num_hidden)
            cell = rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)

            outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32,
                                             sequence_length=self.seqlen)

        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_hidden]
        # if cell is bi-directional, then change back dimension to [batch_size, n_step, n_hidden*2]
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        # 获取按batch_size展开后，每一个sample的最后seq序列
        # 例如：seqlen = [5, 7, 9, 11], seq_max_len = 20, 则 index = [4 26 48 70]
        index = tf.range(0, batch_size) * self.seq_max_len + (self.seqlen - 1)

        if self.conf.use_bi_directional:
            # Indexing
            outputs = tf.gather(tf.reshape(outputs, [-1, self.conf.num_hidden * 2]), index)
            # Linear activation, using outputs computed above
            return tf.matmul(outputs, self.bi_weights['out']) + self.biases['out']
        else:
            outputs = tf.gather(tf.reshape(outputs, [-1, self.conf.num_hidden]), index)
            return tf.matmul(outputs, self.weights['out']) + self.biases['out']

    def define_operator(self):
        """
        定义算子
        :return: 
        """
        logits = self.cell_operation(self.X)
        y_ = tf.nn.softmax(logits)
        predict_pro_op = y_
        predict_op = tf.argmax(y_, 1)
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y))
        optimizer_op = tf.train.GradientDescentOptimizer(learning_rate=self.conf.learning_rate).minimize(loss_op)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(self.Y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return [loss_op, optimizer_op, accuracy_op, predict_op, predict_pro_op]

    def train(self, session):
        """
        模型训练函数
        :param session: 
        :return: 
        """

        training_set = SequenceData(filename=self.conf.training_data, max_seq_len=self.seq_max_len)
        for step in range(1, self.conf.training_steps + 1):
            batch_x, batch_y, batch_seqlen = training_set.next(self.conf.batch_size)
            # Run optimization op (backprop)
            session.run(self.optimizer_op,
                        feed_dict={self.X: batch_x, self.Y: batch_y,
                                   self.seqlen: batch_seqlen,
                                   self.dropout_keep_prob: self.conf.dropout_keep_prob})
            if step % self.conf.display_steps == 0 or step == 1:
                # Calculate batch accuracy & loss
                acc, loss = session.run([self.accuracy_op, self.loss_op],
                                        feed_dict={self.X: batch_x, self.Y: batch_y,
                                                   self.seqlen: batch_seqlen,
                                                   self.dropout_keep_prob: self.conf.dropout_keep_prob})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")
        print('Start to save model.')
        saver = tf.train.Saver()
        saver.save(session, self.conf.save_model_path)

    def test(self, session, load_model=False):
        """
        测试函数
        :param session: 
        :param load_model: 是否加载模型
        :return: 
        """
        if load_model:
            print('Start to load model.')
            saver = tf.train.Saver()
            saver.restore(session, self.conf.load_model_path)
        test_set = SequenceData(filename=self.conf.test_data, max_seq_len=self.seq_max_len)
        test_data = test_set.data
        test_label = test_set.labels
        test_seqlen = test_set.seqlen

        print("Testing Accuracy:", \
              session.run(self.accuracy_op, feed_dict={self.X: test_data, self.Y: test_label,
                                                       self.seqlen: test_seqlen,
                                                       self.dropout_keep_prob: 1.0}))
        print("Testing PR:")
        # 准召结果
        y_list = [label.index(1) for label in test_label]
        predict_pro = session.run(self.predict_pro_op, feed_dict={self.X: test_data,
                                                                  self.seqlen: test_seqlen,
                                                                  self.dropout_keep_prob: 1.0})

        y_pre_list = indicator.threshold_judge_by_y_index(pro_list=predict_pro, axis=1, threshold=0.5)
        indicator.cal_precision_recall_F1(y_list, y_pre_list)

    def predict(self, session, load_model=False):
        """
        预测函数
        :param session: 
        :param load_model: 是否加载模型
        :return: 
        """
        if load_model:
            print('Start to load model.')
            saver = tf.train.Saver()
            saver.restore(session, self.conf.load_model_path)

        predict_set = SequenceData(filename=self.conf.predict_data, max_seq_len=self.seq_max_len)
        predict_result = session.run(self.predict_op, feed_dict={self.X: predict_set.data,
                                                                 self.seqlen: predict_set.seqlen,
                                                                 self.dropout_keep_prob: 1.0})
        predict_result_list = []
        for predict_index in predict_result:
            result = [0] * self.num_class
            result[predict_index] = 1
            predict_result_list.append(result)

        # print("Predict Result:", predict_result)
        print("Predict Result:", predict_result_list)

    def main(self):
        """
        主函数
        :return: 
        """
        with tf.Session() as session:
            init = tf.global_variables_initializer()
            session.run(init)

            self.train(session)
            self.test(session, load_model=True)
            self.predict(session)


if __name__ == '__main__':
    model = DynamicSequenceModel()
    model.main()
