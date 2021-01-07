import tensorflow as tf
import numpy as np
from PIL import Image
from random import shuffle
from keras import backend as K
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2
import os


def dice_loss_with_CE(beta=1, smooth=1e-5):
    def _dice_loss_with_CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))

        tp = K.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        fp = K.sum(y_pred         , axis=[0,1,2]) - tp
        fn = K.sum(y_true[...,:-1], axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        # dice_loss = tf.Print(dice_loss, [dice_loss, CE_loss])
        return CE_loss + dice_loss
    return _dice_loss_with_CE


def CE():
    def _CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))
        # dice_loss = tf.Print(CE_loss, [CE_loss])
        return CE_loss
    return _CE


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def read_label_image_list(data_dir, data_list):
    f = open(data_list, 'r')
    images = []
    masks = []
    lines = []
    for line in f:
        lines.append(line)
    shuffle(lines)
    for line in lines:
        try:
            image, mask = line[:-1].split(' ')
        except ValueError:
            image, mask = line.strip('\n')

        image = os.path.join(data_dir, image)
        mask = os.path.join(data_dir, mask)

        if not tf.gfile.Exists(image):
            raise ValueError('Failed to find file:' + image)
        if not tf.gfile.Exists(mask):
            raise ValueError('Failed to find file:' + mask)

        images.append(image)
        masks.append(mask)
    return images, masks


def letterbox_image(image, label, size):
    label = Image.fromarray(np.array(label))
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    label = label.resize((nw,nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w-nw)//2, (h-nh)//2))
    return new_image, new_label


def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    label = np.float32(label)
    # label = tf.cast(label, dtype=tf.float32)
    label = label - [ignore_label]  # Needs to be subtracted and later added due to 0 padding.
    label = label.reshape((int(label.shape[0]),int(label.shape[1]),-1))
    combined = np.concatenate((image,label),axis=2)
    # combined = tf.concat(axis=2, values=[image, label])
    image_shape = image.shape
    # image_shape = tf.shape(image)
    if crop_w > image_shape[0]:
        combined_pad = np.pad(combined,(((crop_w - image_shape[0])/2,(crop_w - image_shape[0])/2),
                              ((crop_h - image_shape[0])/2,(crop_h - image_shape[0])/2),(0,0)))
    else:
        combined_pad = combined
    # combined_pad = np.pad(combined, ((np.maximum(crop_h,image_shape[0]),np.maximum(crop_h,image_shape[0])),
    #                       (np.maximum(crop_w,image_shape[1]),np.maximum(crop_w,image_shape[1])),(0,0)))
    # combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]),
    #                                             tf.maximum(crop_w, image_shape[1]))
    last_image_dim = image.shape[-1]
    last_label_dim = label.shape[-1]
    # last_image_dim = tf.shape(image)[-1]
    # last_label_dim = tf.shape(label)[-1]

    combined_crop = combined_pad[:, :, :4]
    # combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = np.uint8(label_crop)
    # label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    # img_crop.set_shape((crop_h, crop_w, 3))
    img_crop.reshape((crop_h, crop_w, 3))
    label_crop.reshape((crop_h, crop_w, 1))
    # label_crop.set_shape((crop_h, crop_w, 1))
    return img_crop, label_crop


class Generator(object):
    def __init__(self, data_dir, data_list, batch_size, input_size, ignore_label, img_mean, aux_branch, num_classes):
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.img_mean = img_mean
        self.ignore_label = ignore_label
        self.batch_size = batch_size
        self.aux_branch = aux_branch
        self.num_classes = num_classes
        self.image_list, self.label_list = read_label_image_list(self.data_dir, self.data_list)

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        label = Image.fromarray(np.array(label))

        h, w = input_shape
        # resize image
        rand_jit1 = rand(1-jitter,1+jitter)
        rand_jit2 = rand(1-jitter,1+jitter)
        new_ar = w/h * rand_jit1/rand_jit2

        scale = rand(0.5, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        label = label.convert("L")

        # flip image or not
        flip = rand()<.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        return image_data,label

    def generate(self,random_data=True):
        i = 0
        length = len(self.image_list)
        inputs = []
        targets = []
        h = self.input_size[0]
        w = self.input_size[1]

        while True:
            T = []
            L = []
            img = Image.open(self.image_list[i])
            label = Image.open(self.label_list[i])

            if random_data:
                img, label = self.get_random_data(img, label,(int(self.input_size[1]),int(self.input_size[0])))
            else:
                img, label = letterbox_image(img, label,(int(self.input_size[1]),int(self.input_size[0])))

            # img, label = random_crop_and_pad_image_and_labels(img, label, h, w, self.ignore_label)
            # label = np.squeeze(label,axis=2)
            # seg_label = label.reshape([-1])
            # for i in range(len(seg_label)):
            #     if seg_label[i] < self.num_classes-1:
            #         bool1 = False
            #     else:
            #         bool1 = True
            #     T.append(bool1)
            # indices = np.squeeze(np.where(np.array(T)==True))
            #
            # for j in range(len(indices)):
            #     value = seg_label[indices[j]]
            #     L.append(value)
            #
            # label = np.float32(L)

            png = np.array(label)
            png[png >= self.num_classes] = self.num_classes
            seg_labels = np.eye(self.num_classes+1)[png.reshape([-1])]
            label = seg_labels.reshape((int(w),int(h),self.num_classes+1))
            inputs.append(np.array(img)/255)
            targets.append(label)
            i = (i + 1) % length

            if len(targets) == self.batch_size:
                tmp_inp = np.array(inputs)
                tmp_targets = np.array(targets)
                inputs = []
                targets = []
                if self.aux_branch:
                    yield tmp_inp, np.array([tmp_targets, tmp_targets])
                else:
                    yield tmp_inp, tmp_targets

