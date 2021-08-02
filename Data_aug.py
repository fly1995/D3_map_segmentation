import  numpy as np
import tensorflow as tf
from keras.preprocessing.image import array_to_img


def data_aug(data):
    data_rot90 = tf.image.rot90(data)
    data_rot180 = tf.image.rot90(data_rot90)
    data_rot270 = tf.image.rot90(data_rot180)
    data_u2d = tf.image.flip_up_down(data)
    data_l2r = tf.image.flip_left_right(data)
    data_trans = tf.image.transpose_image(data)
    session = tf.Session()
    data_rot90 = session.run(data_rot90)
    data_rot180 = session.run(data_rot180)
    data_rot270 = session.run(data_rot270)
    data_u2d = session.run(data_u2d)
    data_l2r = session.run(data_l2r)
    data_trans = session.run(data_trans)
    data_flip_rot_save = np.concatenate([data, data_rot90, data_rot180, data_rot270, data_u2d, data_l2r, data_trans], axis=0)
    return data_flip_rot_save


'''
train = np.load(r'E:\D4_map_segmentation\data\es_img.npy')
train = data_aug(train[0:10,:,:,:])
for i in range(train.shape[0]):
    img = train[i, :, :, :]
    img = array_to_img(img)
    img.save(r'E:\D4_map_segmentation\data\test_results\111\%d.jpg' % i)

train_mask = np.load(r'E:\D4_map_segmentation\data\es_map.npy')
train_mask = data_aug(train[0:10,:,:,:])
train_mask = np.argmax(train_mask, axis=-1)
for i in range(train_mask.shape[0]):  # print label, 4 class
    img = train_mask[i, :, :, np.newaxis]
    img = array_to_img(img)
    img.save(r'E:\D4_map_segmentation\data\test_results\111\gt%d.jpg' % i)
'''




