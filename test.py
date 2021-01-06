import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# img_mean = np.array((103.939, 116.779, 123.68))
# ignore_label = 255
# image = tf.read_file('/media/workstation/68bf06e9-e6f8-4333-ae81-6bf0b33b7742/workstation/anaconda3/A-code/3-1PSPNet-tensorflow/data-15/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.jpg')
# image = tf.image.decode_jpeg(image,channels=3)
# image = tf.cast(image,dtype=tf.float32)
# label = Image.open('/media/workstation/68bf06e9-e6f8-4333-ae81-6bf0b33b7742/workstation/anaconda3/A-code/3-1PSPNet-tensorflow/data-15/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds15.png')
# label = np.array(label)
# label = tf.image.decode_png(label, channels=1)
# label = tf.cast(label, dtype=tf.float32)
# label = label - ignore_label
# combined = tf.concat(axis=2, values=[image,label])
# image_shape = tf.shape(image)
# combined_pad = tf.image.pad_to_bounding_box(combined,0,0,tf.maximum(713,1024),tf.maximum(713,2048))
#
# last_image_dim = tf.shape(image)[-1]
# last_label_dim = tf.shape(image)[-1]
# combined_crop = tf.random_crop(combined_pad, [713, 713, 4])
# img_crop = combined_crop[:, :, :last_image_dim]
# label_crop = combined_crop[:, :, last_label_dim:]
# label_crop = label_crop + ignore_label
# label_crop = tf.cast(label_crop, dtype=tf.uint8)
#
# # Set static shape so that tensorflow knows shape at compile time.
# img_crop.set_shape((713, 713, 3))
# label_crop.set_shape((713, 713, 1))
# with tf.Session() as sess:
#     print(sess.run(label_crop))

# a = tf.less_equal((10,5,9,6,2,1,4),(3))
# with tf.Session() as sess:
#     print(sess.run(a))

# label = Image.open('/media/workstation/68bf06e9-e6f8-4333-ae81-6bf0b33b7742/workstation/anaconda3/A-code/3-1PSPNet-tensorflow/data-15/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds15.png')
# label = np.array(label)
# num_classes = 16
# targets = []
# label[label >= num_classes]  = num_classes
# seg_labels = np.eye(num_classes+1)[label.reshape([-1])]
# seg_labels = seg_labels.reshape((int(1024),int(2048),num_classes+1))
#
# targets.append(seg_labels)
# targets.append(seg_labels)
# print(np.shape(targets))

image = [[1,2,3,5,6,7,9],[4,5,6,7,8,9,10]]


# image = Image.open('/media/workstation/68bf06e9-e6f8-4333-ae81-6bf0b33b7742/workstation/anaconda3/A-code/3-1PSPNet-tensorflow/data-15/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.jpg')
#
# iw, ih = image.size
# w, h = 713,713
# scale = min(w/iw, h/ih)
# nw = int(iw*scale)
# nh = int(ih*scale)
#
# images = []
#
# image = image.resize((nw,nh), Image.BICUBIC)
# new_image = Image.new('RGB', (713,713),(128,128,128))
# new_image.paste(image, ((w-nw)//2,(h-nh)//2))
# images.append(new_image)
# print(images)
# print(new_image)

# ROOT_DIR = "/media/workstation/68bf06e9-e6f8-4333-ae81-6bf0b33b7742/workstation/anaconda3/A-code/3-PSPNet-tensorflow-master"
# data_dir = os.path.join(ROOT_DIR, "data-15")
# data_list = os.path.join(ROOT_DIR, "list/cityscapes_train_list.txt")
# input_size = (731,731)
# f = open(data_list, 'r')
# images = []
# masks = []
# for line in f:
#     try:
#         image, mask = line[:-1].split(' ')
#     except ValueError: # Adhoc for test.
#         image = mask = line.strip("\n")
#
#     image = os.path.join(data_dir, image)
#     mask = os.path.join(data_dir, mask)
#
#     if not tf.gfile.Exists(image):
#         raise ValueError('Failed to find file: ' + image)
#
#     if not tf.gfile.Exists(mask):
#         raise ValueError('Failed to find file: ' + mask)
#
#     images.append(image)
#     masks.append(mask)
# images = tf.convert_to_tensor(images, dtype=tf.string)
# labels = tf.convert_to_tensor(masks, dtype=tf.string)
# queue = tf.train.slice_input_producer([images, labels],shuffle=input_size is not None)
# img_contents = tf.read_file(queue[0])
# label_contents = tf.read_file(queue[1])
# print(tf.shape(label_contents))
# label = tf.image.decode_png(label_contents, channels=1)
# print(label)

# label = Image.open('/media/workstation/68bf06e9-e6f8-4333-ae81-6bf0b33b7742/workstation/anaconda3/A-code/3-1PSPNet-tensorflow/data-15/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds15.png')
# image = Image.open('/media/workstation/68bf06e9-e6f8-4333-ae81-6bf0b33b7742/workstation/anaconda3/A-code/3-PSPNet-tensorflow-master/data-15/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.jpg')
# png = np.float32(label)
# image = np.float32(image)
# label = png - 255
# label = label.reshape((int(label.shape[0]),int(label.shape[1]),-1))
# print(label.shape)
# # combined = np.concatenate((image,label),axis=2)
