from nets.PSPNet import pspnet
from PIL import Image
import tensorflow as tf
import numpy as np
import colorsys
import copy
import os

class Pspnet(object):
    _defaults = {
        'model_path'        : 'logs/ep050-loss0.001-val_loss0.004.h5',
        'backbone'          : 'pspnet_resnet50',
        'model_image_size'  : (713,713,3),
        'num_classes'       : 16,
        'downsample_factor' : 16,
        'blend'             : True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    def generate(self):
        self.model = pspnet(self.num_classes,self.model_image_size,downsample_factor=self.downsample_factor,backbone=self.backbone,aux_branch=False)
        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))

        if self.num_classes <= 21:
            self.colors = [(0,0,0),(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128),(0,128,128),(128,128,128),
                           (64,0,0),(192,0,0),(64,128,0),(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128),
                           (0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128),(128,64,12)]

        else:
            hsv_tuples = [(x/len(self.num_names),1,1.)
                          for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x),hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0]*255),int(x[1]*255),int(x[2]*255)),
                    self.colors))

    def letterbox_image(self,image,size):
        iw, ih =image.size
        w, h = size
        scale = min(w/iw,h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh),Image.BICUBIC)
        new_image = Image.new('RGB',size,(128,128,128))
        new_image.paste(image,((w-nw)//2,(h-nh)//2))
        return new_image, nw,nh

    @tf.function
    def get_pred(self,photo):
        preds = self.model(photo,training=False)
        return preds

    def detect_image(self, image):
        old_img = copy.deepcopy(image)
        orinial_h = np.array(image).shape[0]
        orinial_w = np.array(image).shape[1]

        img,nw,nh = self.letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        img = [np.array(img)/255]
        img = np.asarray(img)

        pr = np.array(self.get_pred(img)[0])
        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0],self.model_image_size[1]])
        pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh),int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]

        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))

        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')

        image = Image.fromarray(np.uint8(seg_img)).resize((orinial_w,orinial_h),Image.NEAREST)

        if self.blend:
            image = Image.blend(old_img,image,0.7)

        return image
