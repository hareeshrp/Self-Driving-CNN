import csv  
import cv2
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Dense, Dropout, Flatten, Conv2D, Cropping2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Global Parameters
epochs = 5
batch_size = 32
validation_split = 0.2
correction = 0.2

lines = []
with open("training_data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


def generator(samples, batch_size):
    num_samples = len(samples)
    while True:  
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):  # Loop for center, left, and right images
                    source_path = batch_sample[i]
                    filename = source_path.split("/")[-1]
                    current_path = "training_data/IMG/" + filename
                    image = cv2.imread(current_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
                    images.append(image)

                # Adjusted steering measurements for the side camera images
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                measurements.extend([steering_center, steering_left, steering_right])

            # Data augmentation (flipping images)
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement)
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield shuffle(X_train, y_train)

train_samples, validation_samples = train_test_split(lines, test_size=validation_split)
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential([
    Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)),  
    Cropping2D(cropping=((75, 25), (0, 0))), 

    
    Conv2D(24, (5, 5), activation="relu"),
    MaxPooling2D(),
    Dropout(0.5),
    
    Conv2D(36, (5, 5), activation="relu"),
    MaxPooling2D(),
    
    Conv2D(48, (5, 5), activation="relu"),
    MaxPooling2D(),
    
    Conv2D(64, (3, 3), activation="relu"),
    Conv2D(64, (1, 1), activation="relu"),
    MaxPooling2D(),
    Dropout(0.5),

    Flatten(),

    
    Dense(100, activation="relu"),
    Dense(50, activation="relu"),
    Dense(10, activation="relu"),
    Dense(1)  
])


model.summary()

model.compile(loss="mse", optimizer=Adam())

history = model.fit(
    train_generator,
    steps_per_epoch=ceil(len(train_samples) / batch_size),
    validation_data=validation_generator,
    validation_steps=ceil(len(validation_samples) / batch_size),
    epochs=epochs,
    verbose=1,
)

model.save("model.h5")

# Plot Training Loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Mean Squared Error Loss")
plt.ylabel("Mean Squared Error Loss")
plt.xlabel("Epoch")
plt.legend(loc="upper right")
plt.savefig("examples/mean_squared_error_loss.png")
plt.show()
