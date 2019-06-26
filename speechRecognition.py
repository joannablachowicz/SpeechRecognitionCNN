import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

DATA_PATH = "C:/Users/Joanna/PycharmProjects/SpeechRecognitionCNN/data"


# Input: path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


#convert file to mel-frequency cepstral coefficients
def wav2mfcc(file_path, n_mfcc=20, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=n_mfcc)

    #pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    #cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc

#save wav file to array
def save_data_to_array(path=DATA_PATH, max_len=11, n_mfcc=20):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len, n_mfcc=n_mfcc)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)

# split data
def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)


def prepare_dataset(path=DATA_PATH):
    labels, _, _ = get_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            # Downsampling
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data


def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]

#prepare data

max_len = 11
buckets = 20
channels = 1
epochs = 50
batch_size = 100
num_classes = 8

# Save data to array
save_data_to_array(DATA_PATH, max_len, buckets)
labels=["bed", "bird", "cat", "down", "eight", "five", "four", "go"]
#split data
data_train, data_test, target_train, target_test = get_train_test()
data_train = data_train.reshape(data_train.shape[0], buckets, max_len, channels)
data_test = data_test.reshape(data_test.shape[0], buckets, max_len, channels)
target_train = keras.utils.to_categorical(target_train, num_classes)
target_test = keras.utils.to_categorical(target_test, num_classes)
data_train = data_train.astype('float32')
data_test = data_test.astype('float32')
data_train /= 255
data_test /= 255

#model
convNN = Sequential()
convNN.add(Conv2D(32, (3, 3), strides = (1, 1), padding='same', input_shape=(20, 11, 1), activation='relu'))
convNN.add(Activation('relu'))
convNN.add(Conv2D(32, (3, 3), strides = (3, 3)))
convNN.add(Activation('relu'))
convNN.add(MaxPooling2D(pool_size=(2, 2)))
convNN.add(Dropout(0.25))
convNN.add(Flatten())
convNN.add(Dense(50))
convNN.add(Activation('relu'))
convNN.add(Dense(num_classes))
convNN.add(Activation('softmax'))
convNN.summary()
epochs = 50
batch_size = 300
opt_sgd = keras.optimizers.sgd(lr=0.05)
convNN.compile(loss='categorical_crossentropy', optimizer=opt_sgd, metrics=['accuracy'])
run_hist_sgd = convNN.fit(data_train, target_train, batch_size=batch_size, epochs=epochs,
                          validation_data=(data_test, target_test), shuffle=True,
                          verbose=1)
scores = convNN.evaluate(data_test, target_test, verbose=True)
print(scores)
print('Test accuracy:', scores[1])

