import os
import csv

samples = []
basepath = "./dataset/"
with open(basepath + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
batch_size = 64

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
# Disable SSE warnings from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras import backend as k

# batch_samples = samples[0:2]

# images = []
# angles = []
# for batch_sample in batch_samples:
#     name = basepath+'IMG/'+batch_sample[0].split('\\')[-1]
#     print(name)
#     center_image = cv2.imread(name)
#     center_angle = float(batch_sample[3])
#     images.append(center_image)
#     angles.append(center_angle)

# # trim image to only see section with road
# X_train = np.array(images)
# y_train = np.array(angles)

# print(X_train)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = basepath+'IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # Convert list to numpy array
            X_train = np.array(images)
            y_train = np.array(angles)

            # Reshape images for channels_first data format
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[3], X_train.shape[1], X_train.shape[2])

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 80, 320  # Trimmed image format


# Check the backend and image data format
# If the channel info is to be stored first or last
k.set_image_data_format('channels_first')

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(3, 160,320)))
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))
model.add(Conv2D(16,(3,3), activation='relu', input_shape=(3, row,col)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=2)

# X_train, y_train = (next(train_generator))
# print(X_train.shape)