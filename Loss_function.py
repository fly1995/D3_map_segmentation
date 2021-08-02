#!/usr/bin/python
#coding:utf-8
import numpy as np
from keras import backend as K
import tensorflow as tf
import math
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import jieba



def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) /(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true,y_pred):
    return (1-(dice_coef(y_true, y_pred)))


def jaccard(y_true, y_pred):
    smooth = 1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth)/(K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def Precision(predicted, actual):
    TP = tf.count_nonzero(predicted * actual)
    FP = tf.count_nonzero(predicted * (actual - 1))
    tf_precision = (TP+1) / (TP + FP+1)
    return tf_precision


def Recall(predicted, actual):
    TP = tf.count_nonzero(predicted * actual)
    FN = tf.count_nonzero((predicted - 1) * actual)
    tf_recall = (TP+1) / (TP + FN+1)
    return tf_recall


def F1_score(predicted, actual):
    TP = tf.count_nonzero(predicted * actual)
    FP = tf.count_nonzero(predicted * (actual - 1))
    FN = tf.count_nonzero((predicted - 1) * actual)
    tf_precision = (TP+1) / (TP + FP+1)
    tf_recall =( TP+1) / (TP + FN+1)
    tf_f1_score = 2 * tf_precision * tf_recall / (tf_precision + tf_recall)
    return tf_f1_score


