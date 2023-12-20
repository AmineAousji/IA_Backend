import tensorflow as tf
import keras
from keras.layers import Input, Lambda, Dense, Flatten,Dropout,Conv2D,Rescaling,MaxPooling2D, BatchNormalization
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from keras.regularizers import l2
from tensorflow.python.client import device_lib
import datetime

start_time = datetime.datetime.now()
print("Time of start:", start_time)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#Directory of the dataset for humans and animals
data_dir0 = "D:/DATASETS/Datasets_tests"
data_dir1= "D:/DATASETS/Datasets_vals"
#Directory of the dataset for animals
data_dir= "D:/DATASETS/animals"
data_dir2= "D:/DATASETS/val_animals"
batch_size = 16
img_height = 180
img_width = 180
import matplotlib.pyplot as plt

val_ds=keras.utils.image_dataset_from_directory(data_dir1,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
train_ds =keras.utils.image_dataset_from_directory(data_dir0,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)



num_classes = len(class_names)

model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
  BatchNormalization(),
  MaxPooling2D(2, 2),
  Dropout(0.2),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D(2, 2),
  Flatten(),
  Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
  Dropout(0.2),
  Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001)) # Assuming it's a classification problem
])

opt = tf.keras.optimizers.RMSprop()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Training the model
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
current_time = datetime.datetime.now()
print("Current time:", current_time)

time_difference = current_time - start_time
minutes_difference = time_difference.total_seconds() / 60

print("Time difference in minutes:", minutes_difference)



plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
model.save("./human_animal/hora.keras")


test_img_path= "C:/Users/admin/Desktop/AI_TEST/Lions.jpg"

img = tf.keras.utils.load_img(
    test_img_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)