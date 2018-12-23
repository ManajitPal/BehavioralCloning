import csv
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, Dropout
from keras.layers.convolutional import Cropping2D
from keras.optimizers import Adam

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3): # center, left and rights images
                    name = 'data/IMG/' + batch_sample[i].split('/')[-1]
                    current_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                    images.append(current_image)
                    
                    center_angle = float(batch_sample[3])
                    if i == 0:
                        angles.append(center_angle)
                    elif i == 1: 
                        angles.append(center_angle + 0.4)
                    elif i == 2: 
                        angles.append(center_angle - 0.4)
                    
                    images.append(cv2.flip(current_image, 1))
                    if i == 0:
                        angles.append(center_angle * -1.0)
                    elif i == 1: 
                        angles.append((center_angle + 0.4) * -1.0)
                    elif i == 2: 
                        angles.append((center_angle - 0.4) * -1.0)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# nVidia model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24,(5,5), strides=(2,2), activation='elu'))
model.add(Conv2D(36,(5,5), strides=(2,2), activation='elu'))
model.add(Conv2D(48,(5,5), strides=(2,2), activation='elu'))
model.add(Conv2D(64,(3,3), activation='elu'))
model.add(Conv2D(64,(3,3), activation='elu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit the model
model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)

model.save('model.h5')