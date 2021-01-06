from PSPNet import Pspnet
from PIL import Image
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

pspnet = Pspnet()

while True:
    img = '/media/workstation/68bf06e9-e6f8-4333-ae81-6bf0b33b7742/workstation/anaconda3/A-code/3-1PSPNet-tensorflow/data-15/leftImg8bit/test/berlin/berlin_000002_000019_leftImg8bit.png'
    try:
        image = Image.open(img)
    except:
        print('Open Error ! Try again')
        continue
    else:
        r_image = pspnet.detect_image(image)
        r_image.show()