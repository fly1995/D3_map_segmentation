from Network import *
from Loss_function import *
import numpy as np
from Loss_function import *
import tensorflow as tf
from Metrics import *
from keras.preprocessing.image import array_to_img
session = tf.Session()
import time
time_start = time.time()

def normolize(input):
    epsilon = 1e-6
    mean = np.mean(input)
    std = np.std(input)
    return (input-mean)/(std+epsilon)

img = np.load('E:\D1_Paper_Cardiac_Segmentation\External_validation\\test_data\\t2.npy')#(102, 256, 256, 1) 不加map
map = np.load('E:\\D1_Paper_Cardiac_Segmentation\\External_validation\\map_results\\U5\\c0_map\\c0_map_result.npy')#(102, 256, 256, 1)
img = normolize(img*map)#important
label = np.load('E:\D1_Paper_Cardiac_Segmentation\External_validation\\test_data\\label.npy')#(102, 256, 256, 6)
label = label.astype('float64')
model = U5()
model.load_weights('E:\D1_Paper_Cardiac_Segmentation\code\\(t2)U5_wm.hdf5')
preds4 = model.predict(img)
#np.save(r'E:\D1_Paper_Cardiac_Segmentation\External_validation\\test_results\U5+U5\\t2\\t2_result.npy',preds4)

time_end = time.time()
print('totally cost', time_end-time_start)

def test_npy(file_dir,save_dir):
    npy = np.load(file_dir)
    npy = np.argmax(npy, axis=-1)
    npy = np.expand_dims(npy, axis=-1)
    for i in range(npy.shape[0]):
      img = npy[i,:,:,:]
      img = array_to_img(img)
      img.save(save_dir+'patient%d.jpg'%i)
file_dir=r'E:\D1_Paper_Cardiac_Segmentation\External_validation\\test_results\\U5+U5\\t2\\t2_result.npy'
save_dir=r'E:\D1_Paper_Cardiac_Segmentation\External_validation\\test_results\\U5+U5\\t2\\'
test_npy(file_dir, save_dir)
preds = preds4.astype('float64')
preds[preds>=0.5]=1
preds[preds<0.5]=0
gt1 = label[:, :, :, 1:2]#myo,lv,rv
pred1 = preds[:, :, :, 1:2]#myo,lv,rv
gt2 = label[:, :, :, 2:3]#myo,lv,rv
pred2 = preds[:, :, :, 2:3]#myo,lv,rv
gt3 = label[:, :, :, 3:4]#myo,lv,rv
pred3 = preds[:, :, :, 3:4]#myo,lv,rv
Extrenal_metric(gt1, pred1)
Extrenal_metric(gt2, pred2)
Extrenal_metric(gt3, pred3)


