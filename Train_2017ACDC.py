from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import keras.backend as K
from Network import *
import numpy as np
from Data_aug import *
import time
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
K.set_image_data_format('channels_last')
time_start = time.time()


def normolize(input):
    epsilon = 1e-6
    mean = np.mean(input)
    std = np.std(input)
    return (input-mean)/(std+epsilon)

def train():

    train= np.load('train_img.npy')
    train_mask = np.load('train_mask.npy')

    earlystop = EarlyStopping(monitor='dice_coef', patience=40, verbose=1, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='dice_coef', factor=0.1, patience=20, mode='max')
    #model = MDFA_Net()
    #model = U_Net()
    #model = MSCMR()
    #model = fcn()
    model = path1
    #model = path2()
    #model = Lambda1()
    #model = D_UNet_pp()
    #model = UNet_pp()
    #model = U5()

    #model.load_weights(r'E:\D3_map_segmentation\code\MDFA_Net.hdf5')
    csv_logger = CSVLogger('test.csv')
    model_checkpoint = ModelCheckpoint(filepath='test.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit(train, train_mask, batch_size=6, validation_split=0.1, epochs=1000, verbose=1, shuffle=True,
              callbacks=[model_checkpoint, csv_logger, reduce_lr, earlystop])


if __name__ == '__main__':
    train()

time_end = time.time()
print('totally cost', time_end-time_start)