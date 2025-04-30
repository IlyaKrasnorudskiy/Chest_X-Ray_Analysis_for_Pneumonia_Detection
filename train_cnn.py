import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report

# Проверка и настройка GPU (если есть)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs found:", gpus)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU.")

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                if not img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"Skipping non-image file: {img_path}")
                    continue
                img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_arr is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return data

# Загрузка данных (укажите свои пути)
train = get_training_data('chest_xray/train')
val = get_training_data('chest_xray/val')
test = get_training_data('chest_xray/test')

# Визуализация распределения классов
l = ['Pneumonia' if i[1] == 0 else 'Normal' for i in train]
sns.set_style('darkgrid')
plt.figure(figsize=(6,4))
sns.countplot(l)
plt.title('Class distribution in training set')
plt.show()

# Визуализация примеров
plt.figure(figsize=(5,5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]])
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(labels[train[-1][1]])
plt.show()

# Разделение на признаки и метки
x_train = np.array([item[0] for item in train]) / 255.0
y_train = np.array([item[1] for item in train])

x_val = np.array([item[0] for item in val]) / 255.0
y_val = np.array([item[1] for item in val])

x_test = np.array([item[0] for item in test]) / 255.0
y_test = np.array([item[1] for item in test])

# Изменение формы для CNN
x_train = x_train.reshape(-1, img_size, img_size, 1)
x_val = x_val.reshape(-1, img_size, img_size, 1)
x_test = x_test.reshape(-1, img_size, img_size, 1)

# Аугментация данных
datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)
datagen.fit(x_train)

# Создание модели
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(img_size, img_size, 1)),
    BatchNormalization(),
    MaxPool2D((2,2), padding='same'),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    Dropout(0.1),
    BatchNormalization(),
    MaxPool2D((2,2), padding='same'),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D((2,2), padding='same'),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    Dropout(0.2),
    BatchNormalization(),
    MaxPool2D((2,2), padding='same'),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    Dropout(0.2),
    BatchNormalization(),
    MaxPool2D((2,2), padding='same'),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Коллбек для уменьшения learning rate при застое
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=1e-6)

# Обучение модели
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=12,
    validation_data=datagen.flow(x_val, y_val),
    callbacks=[learning_rate_reduction]
)

# Оценка модели
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Предсказания (замена устаревшего predict_classes)
predictions = (model.predict(x_test) > 0.5).astype("int32").reshape(-1)
print(classification_report(y_test, predictions, target_names=['Pneumonia (Class 0)', 'Normal (Class 1)']))

# Сохранение модели
model.save('pneumonia_cnn_model.h5')

# Визуализация результатов обучения
epochs = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(epochs, history.history['accuracy'], 'go-', label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'ro-', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, history.history['loss'], 'go-', label='Training Loss')
plt.plot(epochs, history.history['val_loss'], 'ro-', label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
