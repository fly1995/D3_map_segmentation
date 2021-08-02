import os
import nibabel as nb
import SimpleITK as sitk
import numpy as np
from Map_Network import *
from Get_weighted_map import *
from keras.preprocessing.image import array_to_img
import time
time_start = time.time()


def predict(root, predict_save):
    file_name1 = os.listdir(root)

    for file in file_name1:
        file_name2 = os.listdir(os.path.join(root, file))

        ED = nb.load(os.path.join(root, file, str(file_name2[2]))).get_data()#(232, 256, 10) file_name2[2]:ED file_name2[3]:ES
        ED_CROP = np.zeros((512, 512, ED.shape[2]))
        ED_CROP[int((512 - ED.shape[0]) / 2):int((512 - ED.shape[0]) / 2 + ED.shape[0]),int((512 - ED.shape[1]) / 2):int((512 - ED.shape[1]) / 2 + ED.shape[1]), :] = ED[:, :, :]
        out = ED_CROP[int((ED_CROP.shape[0] - 160) / 2):int((ED_CROP.shape[0] + 160) / 2),int((ED_CROP.shape[1] - 160) / 2):int((ED_CROP.shape[1] + 160) / 2), :]#(160, 160, x)
        ED_CROP = out[:, :, :, np.newaxis].transpose(2, 1, 0, 3)#(x, 160, 160, 1)

        map_model = up2down()
        map_model.load_weights(r'E:\D3_map_segmentation\code\up2down_map.hdf5')
        map = map_model.predict(ED_CROP)#(x, 160, 160, 1)
        map[map >= 0.5] = 1
        map[map < 0.5] = 0

        for i in range(map.shape[0]):  # out.shape[0]
            img = map[i, :, :, :]
            img = array_to_img(img)
            img.save(predict_save+str(file_name2[2][0:10])+'_map%d.jpg' % i)

        weighted_map = ED_CROP*map
        for i in range(weighted_map.shape[0]):
            img = weighted_map[i, :, :, :]
            img = array_to_img(img)
            img.save(predict_save+str(file_name2[2][0:10])+'_weighted_map%d.jpg' % i)
        y = map[:, :, :, 0]
        print(y.shape)
        y = y.astype('uint8')
        image = sitk.GetImageFromArray(y)
        sitk.WriteImage(image, predict_save + str(file_name2[2]))


root = r'E:\D3_map_segmentation\data\testing'
predict_save = r'E:\D3_map_segmentation\data\test_results\up2down\map\ed\\'
predict(root, predict_save)

time_end = time.time()
print('totally cost', time_end-time_start)