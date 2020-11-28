import tensorflow.keras as keras
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras import preprocessing
import pandas as pd


classes=['mask','no_mask']
base_path= '../data/'
model_name='mask_detection_model3'

# Creating the MobileNetV2 Base-Model
base_model = mobilenet_v2.MobileNetV2(
    weights='imagenet', 
    include_top=False,
    input_shape=(224, 224, 3))

# Freezing the Weights of the Base-Model
base_model.trainable = False

# Definition of the Model
model = keras.Sequential()
model.add(base_model)
model.add(keras.layers.AveragePooling2D((7,7)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(classes), activation='softmax'))
model.summary()


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

# Callback for early Stopping
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)


# Creating a Data Generator for Image Augmentation
data_gen = preprocessing.image.ImageDataGenerator(
    preprocessing_function=mobilenet_v2.preprocess_input,
    fill_mode='nearest',
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    validation_split=0.2)

# Creating the Generators for the Training und Validation Data
train_generator = data_gen.flow_from_directory(
        directory=base_path,
        class_mode="categorical",
        classes=classes,
        batch_size=32,
        target_size=(224, 224),
        subset='training')

validation_generator = data_gen.flow_from_directory(
        directory=base_path,
        class_mode="categorical",
        classes=classes,
        batch_size=32,
        target_size=(224, 224),
        subset='validation')

# Trainings the Model
m_hist=model.fit(train_generator, 
          epochs=100,
          verbose=2,
          callbacks=[callback],
          validation_data=validation_generator,
          steps_per_epoch=34
          )

# Save Model and Training History
hist_df = pd.DataFrame(m_hist.history)
hist_df.to_pickle(f'../models/{model_name}_hist_df.pkl')
model.save(f'../models/{model_name}.h5')
