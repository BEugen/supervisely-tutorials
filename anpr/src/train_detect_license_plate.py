import random
import time
import numpy as np
import os
import json
import shutil
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import TensorBoard
from time import time


SOURCE_PATH = '../data/experiment001/license plate/artificial/'
MODEL_PATH = '../data/model_artif/'
TEST_PATHS = "../data/experement1/artificial_test/"
TRAIN_PATHS = "../data/experement1/artificial_train/"

NB_EPOCH = 25
BATCH_SIZE = 20
SPLIT_SIZE = 0.25
VERBOSE = 1
OPTIM = Adam()

def split_sources():
    #if os.listdir(TEST_PATHS + 'ann/'):
    #    os.remove(TEST_PATHS + 'ann/')
    #    os.remove(TEST_PATHS + 'img/')
    #if os.listdir(TRAIN_PATHS + 'ann/'):
     #   os.remove(TRAIN_PATHS + 'ann/')
    #    os.remove(TRAIN_PATHS + 'img/')
    files = os.listdir(SOURCE_PATH + 'img/')
    split_size = len(files) - int(len(files) * SPLIT_SIZE)
    print(split_size)
    train = files[:split_size]
    test = files[split_size:]
    for file in train:
        out_file = os.path.splitext(file)[0].split('.')
        shutil.copy2(SOURCE_PATH + 'img/' + file, TRAIN_PATHS + 'img/' + file)
        shutil.copy2(SOURCE_PATH + 'ann/' + out_file[0] + '.json', TRAIN_PATHS + 'ann/' + out_file[0] + '.json')

    for file in test:
        out_file = os.path.splitext(file)[0].split('.')
        shutil.copy2(SOURCE_PATH + 'img/' + file, TEST_PATHS + 'img/' + file)
        shutil.copy2(SOURCE_PATH + 'ann/' + out_file[0] + '.json', TEST_PATHS + 'ann/' + out_file[0] + '.json')


def load_image(path):
    list_file = os.listdir(path + 'img/')
    random.seed(40)
    random.shuffle(list_file)
    x_data = []
    y_data = []
    for file in list_file:
        flabel = os.path.splitext(file)[0].split('.')
        im = cv2.imread(path + 'img/' + file,  cv2.IMREAD_GRAYSCALE)
        im = img_to_array(im)
        x_data.append(im)
        y_data.append(load_annotation(path + 'ann/' + flabel[0] + '.json'))
    x_data = np.array(x_data, dtype='float')/255.0
    y_data = np.array(y_data)
    return x_data, y_data


def load_annotation(path):
    with open(path) as data_file:
        data = json.load(data_file)

    left = data["objects"][0]["points"]["exterior"][0][0]
    top = data["objects"][0]["points"]["exterior"][0][1]
    right = data["objects"][0]["points"]["exterior"][2][0]
    bottom = data["objects"][0]["points"]["exterior"][2][1]

    return [left, top, right, bottom]


def model_lp():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, padding="same",
                     input_shape=(64, 128, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=2, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, kernel_size=2, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    #model.add(Reshape((-1, 8*16*128)))
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dense(4))
    return model



def main():
    split_sources()
    X_train, Y_train = load_image(TRAIN_PATHS)
    X_test, Y_test = load_image(TEST_PATHS)
    print("check shapes:")
    print("X_train - ", X_train.shape)
    print("Y_train - ", Y_train.shape)
    print("X_test - ", X_test.shape)
    print("Y_test - ", Y_test.shape)
    tensorboard = TensorBoard(log_dir="../logs/{}".format(time()), write_graph=True, write_grads=True, write_images=True,
                              histogram_freq=0)
    # fit
    model = model_lp()
    model.compile(loss='mean_squared_error', optimizer=OPTIM, metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,
                        validation_data=(X_test, Y_test),
                        validation_split=SPLIT_SIZE, callbacks=[tensorboard])

    score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
    print('Test score:', score[0])
    print('Test accuracy', score[1])
    model_json = model.to_json()
    with open("model_lpd.json", "w") as json_file:
        json_file.write(model_json)
        #serialize weights to HDF5
    model.save_weights("model_lpd.h5")

if __name__ == '__main__':
    main()
