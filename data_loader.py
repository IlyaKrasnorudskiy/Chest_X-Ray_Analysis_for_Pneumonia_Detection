import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_training_data(data_dir, img_size=150):
    """
    Загружает и подготавливает данные из указанной директории.
    
    Args:
        data_dir (str): Путь к директории с данными
        img_size (int): Размер изображения
        
    Returns:
        list: Список кортежей (изображение, метка)
    """
    data = []
    labels = ['PNEUMONIA', 'NORMAL']
    
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                if not img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_arr is None:
                    continue
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return data

def prepare_data(train, val, test, img_size=150):
    """
    Подготавливает данные для обучения модели.
    
    Args:
        train (list): Тренировочные данные
        val (list): Валидационные данные
        test (list): Тестовые данные
        img_size (int): Размер изображения
        
    Returns:
        tuple: Подготовленные данные (x_train, y_train, x_val, y_val, x_test, y_test)
    """
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
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def create_data_generators(x_train, y_train, x_val, y_val, batch_size=32):
    """
    Создает генераторы данных для обучения и валидации.
    
    Args:
        x_train (numpy.ndarray): Тренировочные изображения
        y_train (numpy.ndarray): Тренировочные метки
        x_val (numpy.ndarray): Валидационные изображения
        y_val (numpy.ndarray): Валидационные метки
        batch_size (int): Размер батча
        
    Returns:
        tuple: Генераторы данных (train_generator, val_generator)
    """
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )
    
    val_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)
    
    return train_generator, val_generator 