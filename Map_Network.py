from __future__ import division, print_function
from keras.layers import concatenate, Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from Loss_function import *
from keras.layers import Conv2D, Conv2DTranspose
from keras import Model,layers
from keras.layers import Input,Conv2D,BatchNormalization,Activation,Reshape
from keras.layers import Input, average
shape1=(160, 160, 1)


def U_Net(input_size=(160, 160, 1)):
    flt = 64
    inputs = Input(input_size)
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv5)
    up6 = concatenate([Conv2DTranspose(flt * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv6)
    up7 = concatenate([Conv2DTranspose(flt * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv7)
    up8 = concatenate([Conv2DTranspose(flt * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv8)
    up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up9)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef])
    return model


def U5(input_size=(160, 160, 1)):
    flt = 64
    inputs = Input(input_size)
    conv1 = Conv2D(flt, (5, 5), activation='relu', padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(flt, (5, 5), activation='relu', padding='same',kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(flt * 2, (5, 5), activation='relu', padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(flt * 2, (5, 5), activation='relu', padding='same',kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv5)

    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv61 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    conv61 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv61)
    up = concatenate([Conv2DTranspose(flt * 8, (2, 2), strides=(2, 2), padding='same')(conv61), conv5], axis=3)
    conv = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up)
    conv = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv)

    up6 = concatenate([Conv2DTranspose(flt * 8, (2, 2), strides=(2, 2), padding='same')(conv), conv4], axis=3)
    conv6 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv6)
    up7 = concatenate([Conv2DTranspose(flt * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv7)
    up8 = concatenate([Conv2DTranspose(flt * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(flt * 2, (5, 5), activation='relu', padding='same',kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(flt * 2, (5, 5), activation='relu', padding='same',kernel_initializer='he_normal')(conv8)
    up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(flt, (5, 5), activation='relu', padding='same',kernel_initializer='he_normal')(up9)
    conv9 = Conv2D(flt, (5, 5), activation='relu', padding='same',kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef])
    return model


def conv_bn_relu(input_tensor, flt):
    x = Conv2D(flt, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(flt, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def UNet_pp_func(inputs):#Unet
    flt=32
    conv1_1 = conv_bn_relu(inputs, flt)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    conv2_1 = conv_bn_relu(pool1, flt*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    conv3_1 = conv_bn_relu(pool2, flt*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_1)

    conv4_1 = conv_bn_relu(pool3, flt*8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_1)

    conv5_1 = conv_bn_relu(pool4, flt*16)

    up1_2 = Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv2_1)
    conv1_2 = concatenate([conv1_1, up1_2], 3)
    conv1_2 = conv_bn_relu(conv1_2, flt)

    up2_2 = Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv3_1)
    conv2_2 = concatenate([conv2_1, up2_2], 3)
    conv2_2 = conv_bn_relu(conv2_2, flt*2)

    up3_2 = Conv2DTranspose(flt*4, (2, 2), strides=(2, 2), padding='same')(conv4_1)
    conv3_2 = concatenate([conv3_1, up3_2], 3)
    conv3_2 = conv_bn_relu(conv3_2, flt*4)

    up4_2 = Conv2DTranspose(flt*8, (2, 2), strides=(2, 2), padding='same')(conv5_1)
    conv4_2 = concatenate([conv4_1, up4_2], 3)
    conv4_2 = conv_bn_relu(conv4_2,flt*8)

    up1_3 = Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv2_2)
    conv1_3 = concatenate([conv1_1, conv1_2, up1_3], 3)
    conv1_3 = conv_bn_relu(conv1_3, flt)

    up2_3 = Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv3_2)
    conv2_3 = concatenate([conv2_1, conv2_2, up2_3], 3)
    conv2_3 = conv_bn_relu(conv2_3, flt*2)

    up3_3 = Conv2DTranspose(flt*4, (2, 2), strides=(2, 2), padding='same')(conv4_2)
    conv3_3 = concatenate([conv3_1, conv3_2, up3_3], 3)
    conv3_3 = conv_bn_relu(conv3_3, flt*4)

    up1_4 = Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv2_3)
    conv1_4 = concatenate([conv1_1, conv1_2, conv1_3, up1_4], 3)
    conv1_4 = conv_bn_relu(conv1_4, flt)

    up2_4 = Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv3_3)
    conv2_4 = concatenate([conv2_1, conv2_2, conv2_3, up2_4], 3)
    conv2_4 = conv_bn_relu(conv2_4, flt*2)

    up1_5 = Conv2DTranspose(flt*4, (2, 2), strides=(2, 2), padding='same')(conv2_4)
    conv1_5 = concatenate([conv1_1, conv1_2, conv1_3, conv1_4, up1_5], 3)
    conv1_5 = conv_bn_relu(conv1_5, flt)

    return conv1_5


def UNet_pp(inputs=Input(shape1)):#Unet
    conv1_1 = conv_bn_relu(inputs, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    conv2_1 = conv_bn_relu(pool1, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    conv3_1 = conv_bn_relu(pool2, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_1)

    conv4_1 = conv_bn_relu(pool3, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_1)

    conv5_1 = conv_bn_relu(pool4, 512)

    up1_2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2_1)
    conv1_2 = concatenate([conv1_1, up1_2], 3)
    conv1_2 = conv_bn_relu(conv1_2, 32)

    up2_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3_1)
    conv2_2 = concatenate([conv2_1, up2_2], 3)
    conv2_2 = conv_bn_relu(conv2_2, 64)

    up3_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4_1)
    conv3_2 = concatenate([conv3_1, up3_2], 3)
    conv3_2 = conv_bn_relu(conv3_2, 128)

    up4_2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5_1)
    conv4_2 = concatenate([conv4_1, up4_2], 3)
    conv4_2 = conv_bn_relu(conv4_2, 256)

    up1_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2_2)
    conv1_3 = concatenate([conv1_1, conv1_2, up1_3], 3)
    conv1_3 = conv_bn_relu(conv1_3, 32)

    up2_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3_2)
    conv2_3 = concatenate([conv2_1, conv2_2, up2_3], 3)
    conv2_3 = conv_bn_relu(conv2_3, 64)

    up3_3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4_2)
    conv3_3 = concatenate([conv3_1, conv3_2, up3_3], 3)
    conv3_3 = conv_bn_relu(conv3_3, 128)

    up1_4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2_3)
    conv1_4 = concatenate([conv1_1, conv1_2, conv1_3, up1_4], 3)
    conv1_4 = conv_bn_relu(conv1_4, 32)

    up2_4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3_3)
    conv2_4 = concatenate([conv2_1, conv2_2, conv2_3, up2_4], 3)
    conv2_4 = conv_bn_relu(conv2_4, 64)

    up1_5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv2_4)
    conv1_5 = concatenate([conv1_1, conv1_2, conv1_3, conv1_4, up1_5], 3)
    conv1_5 = conv_bn_relu(conv1_5, 32)

    output = Conv2D(1, (1, 1), activation='sigmoid',)(conv1_5)
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef])
    model.summary()
    return model


def D_UNet_pp(inputs=Input(shape1)):
    p1 = UNet_pp_func(inputs)
    p2 = UNet_pp_func(inputs)
    merge = concatenate([p1, p2], axis=-1)
    output = Conv2D(32, (1, 1), activation='relu', )(merge)
    output = Conv2D(1, (1, 1), activation='sigmoid',)(output)
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef])
    model.summary()
    return model

def up2down(input_size=(160, 160, 1)):
    flt = 32
    inputs = Input(input_size)
    conv11 = Conv2D(flt, (3, 3), dilation_rate=1, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv12 = Conv2D(flt, (3, 3), dilation_rate=2, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv13 = Conv2D(flt, (3, 3), dilation_rate=3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv14 = Conv2D(flt, (3, 3), dilation_rate=4, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv15 = Conv2D(flt, (3, 3), dilation_rate=5, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv11 = Activation('relu')(BatchNormalization()(conv11))
    conv12 = Activation('relu')(BatchNormalization()(conv12))
    conv13 = Activation('relu')(BatchNormalization()(conv13))
    conv14 = Activation('relu')(BatchNormalization()(conv14))
    conv15 = Activation('relu')(BatchNormalization()(conv15))

    concate1 = concatenate([conv11, conv12, conv13, conv14, conv15], axis=-1)
    concate1 = concatenate([Conv2D(flt, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(concate1), inputs], axis=-1)
    concate1 = Activation('relu')(BatchNormalization()(concate1))

    up1 = Conv2DTranspose(flt, (5, 5), dilation_rate=1, activation='relu', padding='valid', kernel_initializer='he_normal')(concate1)
    up1 = Activation('relu')(BatchNormalization()(up1))
    up2 = Conv2DTranspose(flt*2, (5, 5), dilation_rate=2, activation='relu', padding='valid', kernel_initializer='he_normal')(up1)
    up2 = Activation('relu')(BatchNormalization()(up2))
    up3 = Conv2DTranspose(flt*3, (5, 5), dilation_rate=3, activation='relu', padding='valid', kernel_initializer='he_normal')(up2)
    up3 = Activation('relu')(BatchNormalization()(up3))
    up4 = Conv2DTranspose(flt*4, (5, 5), dilation_rate=4, activation='relu', padding='valid', kernel_initializer='he_normal')(up3)
    up4 = Activation('relu')(BatchNormalization()(up4))

    conv1 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up4)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#100
    conv2 = Conv2D(flt * 6, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#50
    conv3 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#25

    u1 = concatenate([Conv2DTranspose(flt * 8, (2, 2), strides=(2, 2), padding='same')(pool3), pool2], axis=-1) #50
    u2 = concatenate([Conv2DTranspose(flt * 6, (2, 2), strides=(2, 2), padding='same')(u1), pool1], axis=-1) #100
    u3 = concatenate([Conv2DTranspose(flt * 4, (2, 2), strides=(2, 2), padding='same')(u2), up4], axis=-1)   #200

    down2 = Conv2D(flt*4, (5, 5), dilation_rate=4, activation='relu', padding='valid', kernel_initializer='he_normal')(u3)
    down2 = concatenate([Activation('relu')(BatchNormalization()(down2)), up3], axis=-1)
    down3 = Conv2D(flt*3, (5, 5), dilation_rate=3, activation='relu', padding='valid', kernel_initializer='he_normal')(down2)
    down3 = concatenate([Activation('relu')(BatchNormalization()(down3)), up2], axis=-1)
    down4 = Conv2D(flt*2, (5, 5), dilation_rate=2, activation='relu', padding='valid', kernel_initializer='he_normal')(down3)
    down4 = concatenate([Activation('relu')(BatchNormalization()(down4)), up1], axis=-1)
    down5 = Conv2D(flt, (5, 5), dilation_rate=1, activation='relu', padding='valid', kernel_initializer='he_normal')(down4)
    down5 = concatenate([Activation('relu')(BatchNormalization()(down5)), concate1], axis=-1)

    conv3 = Conv2D(flt, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(down5)
    output = Conv2D(1, (1, 1), activation='sigmoid')(conv3)
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(lr=0.001), loss=[dice_coef_loss], metrics=[dice_coef])
    model.summary()
    return model