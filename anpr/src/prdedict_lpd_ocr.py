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

sess = tf.Session()
K.set_session(sess)

MODEL_NAME = "model_lpd"
MODEL_OCR_NAME = "model_lpd_ocr"
OPTIM = Adam()
PATH_IMG = "../data/experement1/artificial_test/img/"
letters = sorted(list({'2', '6', '9', '1', '4', 'A', '7', 'M', 'C', 'Y', '0', 'H', 'B',
                      'X', 'K', 'T', 'O', 'E', '8', '5', '3', 'P'}))
MAX_LEN_PLATE = 8

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret


def get_model(img_w):
    # Input Parameters
    img_h = 64

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    batch_size = 32
    downsample_factor = pool_size ** 2

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
        inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(len(letters), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[MAX_LEN_PLATE], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    loaded_model = load_model(MODEL_OCR_NAME + '.h5', compile=False)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    loaded_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    return load_model


def main():
    json_file = open(MODEL_NAME + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(MODEL_NAME + '.h5')
    loaded_model.compile(loss='mean_squared_error', optimizer=OPTIM, metrics=['accuracy'])
    files = os.listdir(PATH_IMG)
    for file in files:
        im_a = cv2.imread(PATH_IMG + file,  cv2.IMREAD_GRAYSCALE)
        im = np.array(im_a, dtype='float') / 255.0
        im = np.expand_dims(im, axis=2)
        im = np.expand_dims(im, axis=0)
        b = loaded_model.predict(im)[0]
        print(b)
        im_a = cv2.imread(PATH_IMG + file)
        cv2.rectangle(im_a, (int(b[0]), int(b[1])), (int(b[2]),                                                     int(b[3])), (0, 255, 0), 1)
        cv2.imwrite(file + ".jpg", im_a)
        break
    load_model_ocr = get_model(im.shape[2])


if __name__ == '__main__':
    main()
