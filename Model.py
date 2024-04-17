import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Load and visualize images
img_size = 48
base_path = "D:/164 Project/images/images/"
plt.figure(figsize=(15, 10))  # Adjust the figure size as needed

count = 0
for expression in os.listdir(os.path.join(base_path, "train")):
    expression_path = os.path.join(base_path, "train", expression)
    images = os.listdir(expression_path)
    for i in range(min(5, len(images))):  # Show up to 5 images or the max available
        count += 1
        plt.subplot(7, 5, count)
        img = tf.keras.utils.load_img(
            os.path.join(expression_path, images[i]), target_size=(img_size, img_size), color_mode="grayscale"
        )
        plt.imshow(img, cmap="gray")
plt.tight_layout()
plt.show()

# Building data generator
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    os.path.join(base_path, "train"),
    target_size=(56, 56),
    color_mode="grayscale",
    batch_size=128,
    class_mode='categorical',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    os.path.join(base_path, "validation"),
    target_size=(56, 56),
    color_mode="grayscale",
    batch_size=128,
    class_mode='categorical',
    shuffle=False
)

# Defining the CNN Model
model = Sequential([
    Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(56, 56, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (5, 5), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.30),
    Conv2D(512, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(512, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.20),
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.20),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.15),
    Dense(7, activation='softmax')  # Assuming there are 7 classes
])

model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
print(model.summary())

# Training the model
epochs = 40
checkpoint = ModelCheckpoint("model_weights.keras", monitor='val_accuracy', save_best_only=True, mode='max')
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size,
    callbacks=[checkpoint]
)

# Visualize training and validation accuracy and loss
def plot_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(24, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.show()

plot_results(history)
