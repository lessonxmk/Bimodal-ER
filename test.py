import glob
import os
import pickle
import random
import time
import math
import logging
import datetime
import tensorflow as tf

import numpy as np
import librosa
from tqdm import tqdm


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


tf.enable_eager_execution()
# setup_seed(111111)
attention_head = 4
attention_hidden = 32
SEED=987654
MODEL_NAME = 'MACNN_seed{}'.format(SEED)
import features
import model as MODEL

Epochs = 50
BATCH_SIZE = 32
learning_rate = 0.001
T_stride = 2
T_overlop = T_stride / 2
overlapTime = {
    'neutral': 1,
    'happy': 1,
    'sad': 1,
    'angry': 1,
}
FEATURES_TO_USE = 'mfcc'  # {'mfcc' , 'logfbank','fbank','spectrogram','melspectrogram'}
featuresExist = True
impro_or_script = 'impro'
featuresFileName = 'features_{}_{}.pkl'.format(FEATURES_TO_USE, impro_or_script)
toSaveFeatures = True
WAV_PATH = "E:/Test/IEMOCAP/"
RATE = 16000

LABEL = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
}

LABEL_DICT1 = {
    '01': 'neutral',
    # '02': 'frustration',
    # '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    # '06': 'fearful',
    '07': 'happy',  # excitement->happy
    # '08': 'surprised'
}
testList = []


def valid_data_process(valid_files, LABEL_DICT1, RATE, t, val_overlap):
    global testList
    val_dict = {}
    if (val_overlap >= t):
        val_overlap = t / 2
    for i, wav_file in enumerate(tqdm(valid_files)):
        label = str(os.path.basename(wav_file).split('-')[2])
        if (label not in LABEL_DICT1):
            continue
        if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
            continue
        label = LABEL_DICT1[label]
        wav_data, _ = librosa.load(wav_file, sr=RATE)
        X1 = []
        y1 = []
        index = 0
        if (t * RATE >= len(wav_data)):
            continue
        while (index + t * RATE < len(wav_data)):
            X1.append(wav_data[int(index):int(index + t * RATE)])
            y1.append(LABEL[label])
            index += int((t - val_overlap) * RATE)

        X1 = np.array(X1)
        val_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }
        testList.append(os.path.basename(wav_file))
    return val_dict

def test(SEED):
    
    data_dir = '../IEMOCAP'
    valid_files = []
    with open('./IEMOCAP_test_{}.csv'.format(SEED)) as f:
        fr = f.readlines()
        for line in fr:
            valid_files.append(data_dir + '/' + line.split('\t')[2])
    val_dict = valid_data_process(valid_files, LABEL_DICT1, RATE, T_stride, 1.6)
    feature_extractor = features.FeatureExtractor(rate=RATE)
    valid_features_dict = {}
    for _, i in enumerate(val_dict):
        X1 = feature_extractor.get_features(FEATURES_TO_USE, val_dict[i]['X'])
        # X1 = tf.expand_dims(X1, 1)
        valid_features_dict[i] = {
            'X': X1,
            'y': val_dict[i]['y']
        }
    model = MODEL.MACNN()
    model.load_weights('./models/{}'.format(MODEL_NAME))

    result = []
    correct = 0
    for _, i in enumerate(valid_features_dict):
        x, y = valid_features_dict[i]['X'], valid_features_dict[i]['y']
        x = tf.expand_dims(x, -1)
        x = tf.cast(x, tf.float32)
        y = np.array([y[0]])
        out = model(x)
        out = tf.reduce_mean(out, 0, keepdims=True).numpy()
        if(np.argmax(out)==y):
            correct+=1
        result.append(out)
    print(correct)
    result = np.array(result)
    np.save('SERtest_{}.npy'.format(SEED), result)

if __name__ == '__main__':
    test(SEED)
