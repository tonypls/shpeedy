import cv2
import os
import time
import numpy as np
import h5py
import os, random
import matplotlib.pylab as plt
import keras.utils.vis_utils as vutil
from skimage.transform import resize
import tensorflow as tf
import keras.models as models
from keras.optimizers import SGD, Adam, RMSprop
from imgaug import augmenters as iaa
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import MaxPooling2D, UpSampling2D, Conv2D, Conv2DTranspose, ZeroPadding2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda, add, LSTM, TimeDistributed, concatenate
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras import backend as K
from keras.utils import conv_utils
from keras.engine.topology import Layer
from keras.engine import InputSpec
import numpy as np
from keras.layers import LSTM
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input
import matplotlib.animation as animation
import sys

FRAME_H, FRAME_W = 112, 112
TIMESTEPS = 16

sometime = lambda aug: iaa.Sometimes(0.3, aug)
sequence = iaa.Sequential([  # sometime(iaa.GaussianBlur((0, 1.5))), # blur images with a sigma between 0 and 3.0
    # sometime(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))), # sharpen images
    # sometime(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 3.), per_channel=0.5)), # add gaussian noise to images
    sometime(iaa.Dropout((0.0, 0.1))),  # randomly remove up to 10% of the pixels
    sometime(iaa.CoarseDropout((0.0, 0.1), size_percent=(0.01, 0.02), per_channel=0.2)),
    sometime(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
],
    random_order=True  # do all of the above in random order
)


def normalize(image):
    return image - [104.00699, 116.66877, 122.67892]


def augment(image, flip, bright_factor):
    # random disturbances borrowed from IAA
    image = sequence.augment_image(image)

    # random brightness change
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * bright_factor
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    # random flip (vertical axis)
    if flip:
        image = cv2.flip(image, 1)

    return image


class BatchGenerator:
    def __init__(self, file_path, indices, batch_size, timesteps=TIMESTEPS, shuffle=True, jitter=True, norm=True,
                 overlap=False):
        self.file_path = file_path
        self.batch_size = batch_size
        self.timesteps = timesteps

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm

        self.images = sorted(os.listdir(self.file_path + 'images/'))
        self.labels = open(self.file_path + 'train.txt').readlines()

        self.indices = indices

    def get_gen(self):
        num_img = len(self.indices)

        l_bound = 0
        r_bound = self.batch_size if self.batch_size < num_img else num_img

        if self.shuffle: np.random.shuffle(self.indices)

        while True:
            if l_bound == r_bound:
                l_bound = 0
                r_bound = self.batch_size if self.batch_size < num_img else num_img

                if self.shuffle: np.random.shuffle(self.indices)

            # the arrays which hold in the inputs and outputs
            x_batch = np.zeros((r_bound - l_bound, self.timesteps, FRAME_H, FRAME_W, 3))
            y_batch = np.zeros((r_bound - l_bound, 1))
            currt_inst = 0

            for index in self.indices[l_bound:r_bound]:
                # if index > 2*self.timesteps:
                #    index -= np.random.randint(0, self.timesteps)

                # construct each input
                flip = (np.random.random() > 0.5)
                bright_factor = 0.5 + np.random.uniform() * 0.5

                for i in range(self.timesteps):
                    image = cv2.imread(self.file_path + 'images/' + self.images[index - self.timesteps + 1 + i])
                    heigh = image.shape[0]
                    transformed = np.concatenate([np.arange(heigh / 3), np.arange(heigh * 2 / 3, heigh)])
                    #print("hey",type(np.arange(heigh / 3)),type(np.arange(heigh * 2 / 3, heigh)),type((np.arange(heigh / 3), np.arange(heigh * 2 / 3, heigh))))
                    image = image[transformed.astype('int64'), :, :]
                    image = cv2.resize(image.copy(), (FRAME_H, FRAME_W))

                    if self.jitter: image = augment(image, flip, bright_factor)
                    if self.norm:   image = normalize(image)

                    x_batch[currt_inst, i] = image

                # construct each output
                speeds = [float(speed) for speed in self.labels[index - self.timesteps + 1:index + 1]]
                y_batch[currt_inst] = np.mean(speeds)

                currt_inst += 1

            yield x_batch, y_batch

            l_bound = r_bound
            r_bound = r_bound + self.batch_size
            if r_bound > num_img: r_bound = num_img

    def get_size(self):
        return len(self.indices) / self.batch_size