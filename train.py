import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'data/train'
test_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size = (48,48), 
batch_size = 64, color_mode = 'grayscale', class_mode = 'categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size = (48,48), 
batch_size = 64, color_mode = 'grayscale', class_mode = 'categorical')

emotional_model = Sequential()
emotional_model.add(Conv2D(filters=32, kernel_size=5, activation='relu', input_shape=[48, 48, 1]))
emotional_model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
emotional_model.add(MaxPooling2D(pool_size=2, strides=2))
emotional_model.add(Dropout(0.25))
emotional_model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
emotional_model.add(MaxPooling2D(pool_size=2, strides=2))
emotional_model.add(Dropout(0.25))
emotional_model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
emotional_model.add(MaxPooling2D(pool_size=2, strides=2))
emotional_model.add(Dropout(0.25))
emotional_model.add(Flatten())
emotional_model.add(Dense(units=1024, activation='relu'))
emotional_model.add(Dropout(0.5))
emotional_model.add(Dense(7, activation = 'softmax'))

emotional_model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = 0.0001), metrics = ['accuracy'])

emotional_model_info = emotional_model.fit_generator(train_generator, steps_per_epoch = 28709 // 64, epochs = 50, 
validation_data = test_generator, validation_steps = 7178 // 64)

emotional_model.save_weights('emojify.h5')