from Network import *
from Loss_function import *
import numpy as np
from Loss_function import *
import tensorflow as tf
from  Metrics import *
from hausdorff import  hausdorff_distance
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
session = tf.Session()
def normolize(input):
    epsilon = 1e-6
    mean = np.mean(input)
    std = np.std(input)
    return (input-mean)/(std+epsilon)

img = np.load('E:\\D1_Paper_Cardiac_Segmentation\\External_validation\\test_data\\c0.npy')#(102, 256, 256, 1)
img =normolize(img)#important

model = MSCMR()#MSCMR
model.load_weights('E:\\D1_Paper_Cardiac_Segmentation\code\\(c0)MSCMR_map.hdf5')
preds4 = model.predict(img)
preds4[preds4>=0.5]=1
preds4[preds4<0.5]=0
print(preds4.shape)
np.save('E:\\D1_Paper_Cardiac_Segmentation\\External_validation\\map_results\\MSCMR\\c0_map\\c0_map_result.npy',preds4)


def test_npy(file_dir,save_dir):
    npy = np.load(file_dir)
    for i in range(npy.shape[0]):
      img = npy[i,:,:,:]
      img = array_to_img(img)
      img.save(save_dir+'patient%d.jpg'%i)
file_dir='E:\\D1_Paper_Cardiac_Segmentation\\External_validation\\map_results\\MSCMR\\c0_map\\c0_map_result.npy'
save_dir='E:\\D1_Paper_Cardiac_Segmentation\\External_validation\\map_results\\MSCMR\\c0_map\\'
test_npy(file_dir, save_dir)

#preds4=np.load('E:\\D1_Paper_Cardiac_Segmentation\\External_validation\\map_results\\UNet\\c0_map\\c0_map_result.npy')
gt = np.load('E:\D1_Paper_Cardiac_Segmentation\External_validation\\test_data\\label_2.npy')
gt = gt.astype('float32')
preds = preds4.astype('float32')
Extrenal_metric(gt, preds4)