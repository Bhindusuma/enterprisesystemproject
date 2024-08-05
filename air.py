import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


#Define paths to your data
base_dir = r'dataset2'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Create subdirectories for train and test
train_cloudy_dir = os.path.join(train_dir, 'cloudy')
train_rainy_dir = os.path.join(train_dir, 'rainy')
train_sunny_dir = os.path.join(train_dir, 'sunny')
test_cloudy_dir = os.path.join(test_dir, 'cloudy')
test_rainy_dir = os.path.join(test_dir, 'rainy')
test_sunny_dir = os.path.join(test_dir, 'sunny')

os.makedirs(train_cloudy_dir, exist_ok=True)
os.makedirs(train_rainy_dir, exist_ok=True)
os.makedirs(train_sunny_dir, exist_ok=True)
os.makedirs(test_cloudy_dir, exist_ok=True)
os.makedirs(test_rainy_dir, exist_ok=True)
os.makedirs(test_sunny_dir, exist_ok=True)

# Move images to corresponding subdirectories in the train directory
for filename in os.listdir(train_dir):
    if os.path.isfile(os.path.join(train_dir, filename)):
        if 'cloudy' in filename:
            shutil.move(os.path.join(train_dir, filename), os.path.join(train_cloudy_dir, filename))
        elif 'rainy' in filename:
            shutil.move(os.path.join(train_dir, filename), os.path.join(train_rainy_dir, filename))
        elif 'sunny' in filename:
            shutil.move(os.path.join(train_dir, filename), os.path.join(train_sunny_dir, filename))

# Move images to corresponding subdirectories in the test directory
for filename in os.listdir(test_dir):
    if os.path.isfile(os.path.join(test_dir, filename)):
        if 'cloudy' in filename:
            shutil.move(os.path.join(test_dir, filename), os.path.join(test_cloudy_dir, filename))
        elif 'rainy' in filename:
            shutil.move(os.path.join(test_dir, filename), os.path.join(test_rainy_dir, filename))
        elif 'sunny' in filename:
            shutil.move(os.path.join(test_dir, filename), os.path.join(test_sunny_dir, filename))


# Define image size and batch size
IMG_SIZE = 224
BATCH_SIZE = 32

# Paths to your data
train_dir = r"C:/Users/Prave/Downloads/enterprisesystemproject/dataset2/train"
test_dir = r"C:/Users/Prave/Downloads/enterprisesystemproject/dataset2/test"

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load the base model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Modify the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Assuming binary classification
])

# Freeze the base model
base_model.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model
history = model.fit(
    train_generator,
    epochs=30,  # Increase the number of epochs
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Unfreeze some layers of the base model for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Fine-tune the model
history_fine = model.fit(
    train_generator,
    epochs=10,  # Additional epochs for fine-tuning
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {accuracy:.2f}')

model.save('weather_prediction_model.h5')
