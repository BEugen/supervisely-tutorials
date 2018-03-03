import os
import json
import shutil

SPLIT_SIZE = 0.25
SOURCE_PATH = '../data/val/anpr_ocr/train/'
TEST_PATHS = "../data/test/anpr_ocr/test/"

files = os.listdir(SOURCE_PATH + 'img/')
split_size = len(files) - int(len(files) * SPLIT_SIZE)
print(split_size)
train = files[:split_size]
test = files[split_size:]

for file in test:
    out_file = os.path.splitext(file)[0].split('.')
    shutil.copy2(SOURCE_PATH + 'img/' + file, TEST_PATHS + 'img/' + file)
    shutil.copy2(SOURCE_PATH + 'ann/' + out_file[0] + '.json', TEST_PATHS + 'ann/' + out_file[0] + '.json')