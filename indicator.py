#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Filename @ indicator.py
 Author @ huangjunheng
 Create date @ 2017-06-11 10:14:06
 Description @ 
"""
# Script starts from here
import sys


def cal_precision_recall_F1(y=None, y_predict=None):
    """
    计算精确率，召回率 和 F1
    """
    TP, TN, FP, FN = 0, 0, 0, 0
    print >> sys.stderr, 'Test sample size: %d.' % len(y)
    print >> sys.stderr, 'Target sameple size: %d.' % sum(y)
    if len(y) != len(y_predict):
        print >> sys.stderr, 'Error:', 'Length is not equal. y is %d, y_predict is %d.' \
                                       % (len(y), len(y_predict))
        return -1
    for i in range(len(y)):
        if y[i] == y_predict[i] and y[i] == 1:
            TP += 1
        elif y[i] == y_predict[i] and y[i] == 0:
            TN += 1
        elif y[i] != y_predict[i] and y[i] == 1:
            FN += 1
        else:
            FP += 1
    accuracy = float(TP + TN) / (TP + FP + TN + FN)
    precision = float(TP) / (TP + FP + 0.001)  # 避免除零异常
    recall = float(TP) / (TP + FN + 0.001)
    F1 = float(2 * precision * recall) / (precision + recall + 0.001)
    print >> sys.stderr, 'TP: %d, FP: %d, FN: %d, TN: %d.' % (TP, FP, FN, TN)
    print >> sys.stderr, 'Accuracy: %.3f' % (accuracy)
    print >> sys.stderr, 'Precision: %.3f, Recall: %.3f, F1: %.3f.' % (precision, recall, F1)

    return precision, recall, F1


def threshold_judge_by_y_index(pro_list, axis, threshold):
    """
    threshold judge
    :param pro_list: 概率列表，长度为样本个数
    :param threshold: 概率门限
    :param axis: pro_list中pro概率的维度
    :return: 
    """
    y_pre_list = []
    for l in pro_list:
        # 需要验证的那一个维度大于threshold就为1，反之为0
        label = 1 if l[axis] >= threshold else 0
        y_pre_list.append(label)

    return y_pre_list


if __name__ == '__main__':
    cal_precision_recall_F1()