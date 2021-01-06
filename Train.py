from keras.callbacks import TensorBoard,ReduceLROnPlateau,EarlyStopping
from keras.optimizers import Adam
from nets.PSPNet import pspnet
from nets.pspnet_training import Generator, dice_loss_with_CE, CE
from utils.metrics import Iou_score, f_score
# from utils.utils import ModelCheckpoint
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import argparse
import numpy as np
import os

IMG_MEAN = np.array((108.939,116.779,123.68),dtype=np.float32)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

LOG_DIR = 'logs/'

ROOT_DIR = "/media/workstation/68bf06e9-e6f8-4333-ae81-6bf0b33b7742/workstation/anaconda3/A-code/3-1PSPNet-tensorflow"
DATA_DIRECTORY = os.path.join(ROOT_DIR, "data-15")
DATA_LIST_PATH = os.path.join(ROOT_DIR, "list/cityscapes_train_list.txt")
DATA_VAL_LIST_PATH = os.path.join(ROOT_DIR, 'list/cityscapes_val_list.txt')
MODEL_PATH = os.path.join(ROOT_DIR,"model/pspnet_resnet50.h5")
BATCH_SIZE = 8
RESNET_FREEZE = 172
INPUT_SIZE = '713, 713, 3'
NUM_CLASSES = 16
BACKBONE = 'pspnet_resnet50'
SAVE_NUM_IMAGES = 4
SAVE_PRED_EVERY = 100
RANDOM_SEED = 1234
IGNORE_LABEL = 255
dic_loss = False
aux_branch = False
downsample_factor = 16


def get_arguments():
    parser = argparse.ArgumentParser(description='Resnet Network')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help='Number of images sent to the network in on step')
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help='Path to the directory containing the dateset')
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help='Path to the file listing the train images in the dateset.')
    parser.add_argument('--data-val', type=str, default=DATA_VAL_LIST_PATH,
                        help='Path tp the file listing the val images in the dataset')
    parser.add_argument('--ignore-label', type=int, default=IGNORE_LABEL)
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help='Comma-separated string with height and width of images')
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help='Load the pertrained model to train PSPNet')
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help='Number of classes to predict (including background)')
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help='Random seed to have reproducible results')
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Where to save Checkpoint of the model")
    return parser.parse_args()


def main():

    args = get_arguments()

    h, w, c = map(int, args.input_size.split(','))
    input_size = (h, w, c)

    model = pspnet(args.num_classes, input_size, downsample_factor=downsample_factor,backbone=BACKBONE, aux_branch=aux_branch)
    model.summary()

    model.load_weights(args.model_path, by_name=True, skip_mismatch=True)

    tf.set_random_seed(args.random_seed)

    with open('list/cityscapes_train_list.txt', 'r') as f :
        train_lines = f.readlines()
    with open('list/cityscapes_val_list.txt', 'r') as f:
        val_lines = f.readlines()

    checkpoint_period = ModelCheckpoint(args.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                        monitor='val_loss',save_weights_only=True, save_best_only=False,period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    tensorboard = TensorBoard(log_dir=args.log_dir)

    for i in range(RESNET_FREEZE):
        model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(RESNET_FREEZE, len(model.layers),args.batch_size))

    if True:
        lr = 1e-4
        Init_Epoch = 0
        Freeze_Epoch = 50
        BATCH_SIZE = args.batch_size

        # model.compile(loss=dice_loss_with_CE() if dic_loss else CE(),optimizer=Adam(lr=lr),metrics=[f_score()])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

        gen = Generator(args.data_dir,args.data_list,BATCH_SIZE,input_size[:2],args.ignore_label,IMG_MEAN,aux_branch,len(train_lines),args.num_classes).generate()
        gen_val = Generator(args.data_dir,args.data_val,BATCH_SIZE,input_size[:2],args.ignore_label,IMG_MEAN,aux_branch,len(val_lines),args.num_classes).generate(False)
        model.fit_generator(gen,
                            steps_per_epoch=max(1, len(train_lines)//BATCH_SIZE)//4,
                            validation_data=gen_val,
                            validation_steps=max(1, len(val_lines)//BATCH_SIZE)//4,
                            epochs=Freeze_Epoch,
                            initial_epoch=Init_Epoch,
                            callbacks=[checkpoint_period, reduce_lr, tensorboard])

    for i in range(RESNET_FREEZE):
        model.layers[i].trainable = True

    if True:
        lr = 1e-5
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100
        BATCH_SIZE = args.batch_size / 2

        model.compile(loss=dice_loss_with_CE() if dic_loss else CE(),
                      optimizer=Adam(lr=lr),
                      metrics=[f_score()])
        print(
            'Freeze the first {} layers of total {} layers.'.format(RESNET_FREEZE, len(model.layers), args.batch_size))

        gen = Generator(args.data_dir,args.data_list,BATCH_SIZE,input_size[:2],args.ignore_label,IMG_MEAN,aux_branch,len(train_lines),args.num_classes).generate()
        gen_val = Generator(args.data_dir,args.data_val,BATCH_SIZE,input_size[:2],args.ignore_label,IMG_MEAN,aux_branch,len(val_lines),args.num_classes).generate(False)

        model.fit_generator(gen,
                            steps_per_epoch=max(1, len(train_lines) // BATCH_SIZE),
                            validation_data=gen_val,
                            validation_steps=max(1, len(val_lines) // BATCH_SIZE),
                            epochs=Unfreeze_Epoch,
                            initial_epoch=Freeze_Epoch,
                            callbacks=[checkpoint_period, reduce_lr, tensorboard])


if __name__ == '__main__':
    main()

