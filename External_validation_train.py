from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import keras.backend as K
from Network import *
import numpy as np
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
K.set_image_data_format('channels_last')

def normolize(input):
    epsilon = 1e-6
    mean = np.mean(input)
    std = np.std(input)
    return (input-mean)/(std+epsilon)
def train():
    #不加map的情况下迅雷
    '''
    train = np.load('E:\D1_Paper_Cardiac_Segmentation\External_validation\\train_data\processed_data\\天aug.npy')
    train_mask = np.load('E:\D1_Paper_Cardiac_Segmentation\External_validation\\train_data\processed_data\\t2gtaug.npy')
    val = np.load('E:\D1_Paper_Cardiac_Segmentation\External_validation\\test_data\\t2.npy')  # (102, 160,160, 1)
    val_mask = np.load('E:\D1_Paper_Cardiac_Segmentation\External_validation\\test_data\\label.npy')  # (102,  160,160,
    '''
    train = np.load('E:\D1_Paper_Cardiac_Segmentation\External_validation\\train_data\processed_data\\t2_map\\t2aug_map.npy')
    train_mask = np.load('E:\D1_Paper_Cardiac_Segmentation\External_validation\\train_data\processed_data\\t2gtaug.npy')
    train = np.concatenate([train, train], axis=0)
    train_mask = np.concatenate([train_mask, train_mask], axis=0)
    val = np.load('E:\D1_Paper_Cardiac_Segmentation\External_validation\\test_data\\t2_map_label\\t2_map.npy')  # (102, 160,160, 1)
    val_mask = np.load('E:\D1_Paper_Cardiac_Segmentation\External_validation\\test_data\\label.npy')  # (102,  160,160, 6)


    train = normolize(train)
    val = normolize(val)
    earlystop = EarlyStopping(monitor='dice_coef', patience=40, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='dice_coef', factor=0.1, patience=20, mode='auto')
    #model = MDFA_Net()
    #model = U_Net()
    #model = MSCMR()
    #model = fcn()
    #model = NDD_Net1()#path1
    #model = path2()
    #model = Lambda1()
    #model = D_UNet_pp()
    #model = UNet_pp()
    model = U5()
    #model.load_weights('E:\D1_Paper_Cardiac_Segmentation\code\(t2)U5_wm.hdf5')
    csv_logger = CSVLogger('(t2)U5_wm.csv')
    model_checkpoint = ModelCheckpoint(filepath='(t2)U5_wm.hdf5', monitor='loss', verbose=1, save_best_only=True,mode='min')
    model.fit(train, train_mask, batch_size=32, validation_data=(val, val_mask), epochs=1000, verbose=1, shuffle=True,
              callbacks=[model_checkpoint, csv_logger, reduce_lr, earlystop])

if __name__ == '__main__':
    train()

