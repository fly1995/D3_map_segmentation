import numpy as np
from keras.preprocessing.image import array_to_img


def get_map(train,train_mask):
    train_mask = np.argmax(train_mask, axis=-1)#(951, 160, 160, 4)
    train_map = []
    for i in range(train.shape[0]):
        x= train[i,:,:,0]
        y= train_mask[i,:,:]
        x = np.array(x).reshape(160*160)#65536
        y = np.array(y).reshape(160*160)#65536
        label_index = np.where(y>0)#(array([23158, 23159, 23160, ..., 41611, 41612, 41613], dtype=int64),)
        nonlabel_index = np.where(y<=0)#(array([    0,     1,     2, ..., 65533, 65534, 65535], dtype=int64),)
        for j in range(len(nonlabel_index[0])):
            x[nonlabel_index[0][j]]=0
        x[label_index] = 1#可以调整为不同的数值
        out = np.array(x).reshape(160, 160)
        out = np.expand_dims(out, axis=-1)
        train_map.append(out)
    train_map_save = np.concatenate(train_map, axis=-1)
    train_map_save = np.expand_dims(train_map_save, axis=-1)
    train_map_save = train_map_save.transpose(2, 0, 1, 3)
    return train_map_save


train = np.load(r'E:\D4_map_segmentation\data\es_img.npy')
train_mask = np.load(r'E:\D4_map_segmentation\data\es_gt.npy')
out = get_map(train, train_mask)
np.save(r'E:\D4_map_segmentation\data\es_map.npy', out)
for i in range(out.shape[0]):#out.shape[0]
    img = out[i, :, :, :]
    img = array_to_img(img)
    img.save(r'E:\D4_map_segmentation\data\ES\map\patient%d.jpg'%i)

