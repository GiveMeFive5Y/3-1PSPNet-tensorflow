from keras.models import *
from keras.layers import *
from nets.Resnet import get_resnet50_encoder
from nets.Resnet import get_resnet101_encoder
import keras.backend as K
import tensorflow as tf
import numpy as np

IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1


def pool_block(feats, pool_factor, out_channel):
    h = K.int_shape(feats)[1]
    w = K.int_shape(feats)[2]

    pool_size = strides = [int(np.round(float(h) / pool_factor)), int(np.round(float(w) / pool_factor))]

    x = AveragePooling2D(pool_size, data_format=IMAGE_ORDERING, strides=strides, padding='SAME')(feats)
    x = Conv2D(out_channel, (1,1), data_format=IMAGE_ORDERING, padding='SAME', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(lambda x: tf.compat.v1.image.resize_images(x, (K.int_shape(feats)[1], K.int_shape(feats)[2]),
                                                          align_corners=True))(x)
    return x


def pspnet(n_classes, input_size, downsample_factor=8, backbone='pspnet_resnet50', aux_branch=True):
    if backbone == 'pspnet_resnet50':
        img_input, f4, o = get_resnet50_encoder(input_size, downsample_factor=downsample_factor)
        out_channel = 512
    elif backbone == 'pspnet_resnet101':
        img_input, f4, o = get_resnet101_encoder(input_size, downsample_factor=downsample_factor)
        out_channel = 512
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

    pool_factor = [1, 2, 3, 6]
    pool_outs = [o]

    for p in pool_factor:
        pooled = pool_block(o, p, out_channel)
        pool_outs.append(pooled)

    o = Concatenate(axis=MERGE_AXIS)(pool_outs)

    #conv2d_5
    o = Conv2D(out_channel, (3, 3), data_format=IMAGE_ORDERING, padding='SAME', use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Dropout(0.1)(o)
    #conv2d_6
    o = Conv2D(n_classes+1, (1, 1), data_format=IMAGE_ORDERING, padding='SAME')(o)
    o = Lambda(lambda x: tf.compat.v1.image.resize_images(x, (input_size[1], input_size[0]), align_corners=True))(o)
    o = Activation('softmax', name='main')(o)

    if aux_branch:
        f4 = Conv2D(out_channel//2, (3, 3), data_format=IMAGE_ORDERING, padding='SAME', use_bias=False)(f4)
        f4 = BatchNormalization()(f4)
        f4 = Activation('relu')(f4)
        f4 = Dropout(0.1)(f4)

        f4 = Conv2D(n_classes, (1, 1), data_format=IMAGE_ORDERING, padding='SAME')(f4)
        f4 = Lambda(lambda x: tf.compat.v1.image.resize_images(x, (input_size[1], input_size[0]), align_corners=True))(
            f4)
        f4 = Activation('softmax', name='aux')(f4)
        model = Model(img_input, [f4, o])
        return model
    else:
        model = Model(img_input, [o])
        return model
