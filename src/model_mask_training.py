from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras import preprocessing



classes=['mask','no_mask']

base_path= '../data/'

data_gen = preprocessing.image.ImageDataGenerator(
    preprocessing_function=mobilenet_v2.preprocess_input,
    fill_mode='nearest',
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)

train_data_gen = data_gen.flow_from_directory(
        directory=base_path,
        class_mode="categorical",
        classes=classes,
        batch_size=100,
        target_size=(224, 224)
)

xtrain, ytrain = next(train_data_gen)


base_model = mobilenet_v2.MobileNetV2(
    weights='imagenet', 
    alpha=0.35,
    pooling='avg',
    include_top=False,
    input_shape=(224, 224, 3)
)


base_model.trainable = False


model = keras.Sequential()
model.add(base_model)
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(classes), activation='softmax'))
model.summary()


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit(xtrain, ytrain, 
          epochs=100, 
          verbose=2,
          batch_size=len(xtrain), 
          callbacks=[callback],
          validation_split=0.3)



model.save('../models/test.h5')
