from __future__ import print_function
import tensorflow as tf
from keras import layers
from keras import Input
from keras.layers import Dense,Conv2D,MaxPool2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Activation,BatchNormalization,Flatten

def identity_block(input_tensor, kernel_size, filters, stage, block, dilation_rate=1):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1,1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', dilation_rate=dilation_rate, name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1,1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    x = Activation('relu')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2), dilation_rate=1):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1,1), strides=strides,
               name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', dilation_rate=dilation_rate,
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1,1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shorcut = Conv2D(filters3, (1,1), strides=strides,
                     name=conv_name_base + '1', use_bias=False)(input_tensor)
    shorcut = BatchNormalization(name=bn_name_base + '1')(shorcut)

    x = layers.add([x,shorcut])
    x = Activation('relu')(x)
    return x


def get_resnet50_encoder(input_size, downsample_factor=8):
    if downsample_factor == 16:
        block4_dilation = 1
        block5_dilation = 2
        block4_stride = 2
    elif downsample_factor == 8:
        block4_dilation = 2
        block5_dilation = 4
        block4_stride = 1
    else:
        raise ValueError('Unsupported factor - `{}`, Use 8 or 16.'.format(downsample_factor))
    img_input = Input(shape=input_size)

    x = ZeroPadding2D(padding=(1,1), name='conv1_pad')(img_input)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), name='Conv1', use_bias=False)(x)
    x = BatchNormalization(axis=-1, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1,1), name='conv2_pad')(x)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), name='Conv2', use_bias=False)(x)
    x = BatchNormalization(axis=-1, name='bn_conv2')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1,1), name='conv3_pad')(x)
    x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), name='Conv3', use_bias=False)(x)
    x = BatchNormalization(axis=-1, name='bn_conv3')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1,1), name='pool1_pad')(x)
    x = MaxPool2D((3,3), strides=(2,2))(x)

    x = conv_block(x, 3, [64,64,256], stage=2, block='a', strides=(1,1))
    x = identity_block(x, 3, [64,64,256], stage=2, block='b')
    x = identity_block(x, 3, [64,64,256], stage=2, block='c')

    x = conv_block(x, 3, [128,128,512], stage=3, block='a')
    x = identity_block(x, 3, [128,128,512], stage=3, block='b')
    x = identity_block(x, 3, [128,128,512], stage=3, block='c')
    x = identity_block(x, 3, [128,128,512], stage=3, block='d')

    x = conv_block(x, 3, [256,256,1024], stage=4, block='a', strides=(block4_stride, block4_stride))
    x = identity_block(x, 3, [256,256,1024], stage=4, block='b', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='c', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='d', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='e', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='f', dilation_rate=block4_dilation)
    f4 = x

    x = conv_block(x, 3, [512,512,2048], stage=5, block='a', strides=(1,1), dilation_rate=block4_dilation)
    x = identity_block(x, 3, [512,512,2048], stage=5, block='b', dilation_rate=block5_dilation)
    x = identity_block(x, 3, [512,512,2048], stage=5, block='c', dilation_rate=block5_dilation)
    f5 = x

    return img_input, f4, f5

def get_resnet101_encoder(input_size, downsample_factor=8):
    if downsample_factor == 16:
        block4_dilation = 1
        block5_dilation = 2
        block4_stride = 2
    elif downsample_factor == 8:
        block4_dilation = 2
        block5_dilation = 4
        block4_stride = 1
    else:
        raise ValueError('Unsupported factor - `{}`, Use 8 or 16.'.format(downsample_factor))
    img_input = Input(shape=input_size)

    x = ZeroPadding2D(padding=(1,1),name='conv1_pad')(img_input)
    x = Conv2D(filters=64,kernel_size=(3,3), strides=(2,2), name='Conv1', use_bias=False)(x)
    x = BatchNormalization(axis=-1, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1,1),name='conv2_pad')(img_input)
    x = Conv2D(filters=64,kernel_size=(3,3), strides=(2,2), name='Conv2', use_bias=False)(x)
    x = BatchNormalization(axis=-1, name='bn_conv2')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1,1), name='conv3_pad')(x)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), name='Conv3', use_bias=False)(x)
    x = BatchNormalization(axis=-1, name='bn_conv3')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1,1), name='pool1_pad')(x)
    x = MaxPool2D((3,3), strides=(2,2))(x)

    x = conv_block(x,3,[64,64,256],stage=2, block='a', strides=(1,1))
    x = identity_block(x, 3, [64,64,256], stage=2, block='b')
    x = identity_block(x, 3, [64,64,256], stage=2, block='c')

    x = conv_block(x, 3, [128,128,512], stage=3, block='a')
    x = identity_block(x, 3, [128,128,512], stage=3, block='b')
    x = identity_block(x, 3, [128,128,512], stage=3, block='c')
    x = identity_block(x, 3, [128,128,512], stage=3, block='d')

    x = conv_block(x, 3, [256,256,1024], stage=4, block='a', strides=(block4_stride, block4_stride))
    x = identity_block(x, 3, [256,256,1024], stage=4, block='b', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='c', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='d', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='e', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='f', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='g', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='h', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='i', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='j', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='k', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='l', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='m', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='n', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='o', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='p', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='q', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='r', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='s', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='t', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='u', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='v', dilation_rate=block4_dilation)
    x = identity_block(x, 3, [256,256,1024], stage=4, block='w', dilation_rate=block4_dilation)
    f4 = x

    x = conv_block(x, 3, [512,512,2048], stage=5, block='a', strides=(1,1), dilation_rate=block4_dilation)
    x = identity_block(x, 3, [512,512,2048], stage=5, block='b', dilation_rate=block5_dilation)
    x = identity_block(x, 3, [512,512,2048], stage=5, block='c', dilation_rate=block5_dilation)
    f5 = x
    return img_input, f4, f5