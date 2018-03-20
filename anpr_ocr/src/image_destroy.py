import time
import numpy as np
import os
import json
import shutil
import cv2
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import GRU
from time import time
from keras.models import model_from_json
import tensorflow as tf
import itertools




IMG_PATH_SOURCE = '/home/eugen/PycharmProjects/supervisely-tutorials/anpr/data/val/anpr_ocr/train/img'
IMG_PATH_DEST = '/home/eugen/PycharmProjects/supervisely-tutorials/anpr_ocr/data/val/anpr_ocr/train/img'

def main():
    for file in os.listdir(IMG_PATH_SOURCE):
        print(file)
        img = cv2.imread(IMG_PATH_SOURCE + "/" + file)
        img = cv2.cvtColor(cv2.resize(img, (90, 20)), cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (152, 34))
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 1)
        cv2.imwrite(IMG_PATH_DEST + "/" + file, img)


if __name__ == '__main__':
    main()
