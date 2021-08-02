from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import keras.backend as K
from Map_Network import *
import numpy as np
from keras.models import load_model
from skimage import filters, exposure
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
K.set_image_data_format('channels_last')
from Data_aug import *
import time

time_start = time.time()

def train():
    #训练前景和背景ACDC数据集
    train= np.load('train_img.npy')
    train_mask = np.load('train_map.npy')

    earlystop = EarlyStopping(monitor='val_dice_coef', patience=40, verbose=1, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.1, patience=20, mode='max')
    model = D_UNet_pp()
    #model = MDFA_Net1()
    #model = U_Net()
    #model = MSCMR()
    #model = fcn()
    #model = NDD_Net1()
    #model = path2()
    #model = Lambda1()
    #model = UNet_pp()
    #model = U5()
    #model = up2down()
    #model.load_weights(r'E:\D3_map_segmentation\code\U5_map_c0.hdf5')
    csv_logger = CSVLogger('test.csv')
    model_checkpoint = ModelCheckpoint(filepath='test.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit(train, train_mask,  batch_size=12, validation_split=0.1,  epochs=1000, verbose=1, shuffle=True,
              callbacks=[model_checkpoint, csv_logger, reduce_lr, earlystop])


if __name__ == '__main__':
    train()


time_end = time.time()
print('totally cost', time_end-time_start)

