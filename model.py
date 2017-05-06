import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
# Disable SSE warnings from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Reshape
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras import backend as k
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

samples = []
basepath = "./dataset_5/"

# Load the data from CSV files into memory
with open(basepath + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split the data into training set and validation set in 4:1 ratio
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Decide hyperparameters 
batch_size = 128
epochs = 10

# original image dimensions
orow, ocol = 160, 320
input_channels = 3

# Cropping2D parameters
ctop = 50
cbottom = 20
cleft = 0
cright = 0

# Cropped image dimensions
row, col = (orow - (ctop + cbottom)), (ocol - (cleft + cright))

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

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Check the backend and image data format
# set the image data format to channels_first
k.set_image_data_format('channels_first')

model = Sequential()

# Preprocessing 
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(orow, ocol, input_channels), output_shape=(orow, ocol, input_channels)))
model.add(Reshape((input_channels, orow, ocol)))
model.add(Cropping2D(cropping=((ctop,cbottom), (cleft,cright)), input_shape=(input_channels, orow, ocol)))


# Convolution layers 
model.add(Conv2D(24,(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(36,(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(48,(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))

# Flatten 
model.add(Flatten())

# Fully connected layers
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=(len(validation_samples)/batch_size), epochs=epochs)

# Save model
model.save('./model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# X_train, y_train = (next(train_generator))
# print(X_train.shape)