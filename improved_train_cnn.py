import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

# Проверка и настройка GPU
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

# Конфигурация
IMG_SIZE = 224  # Увеличенный размер для лучшего качества
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

def get_training_data(data_dir):
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
                resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                # Преобразование в RGB для использования с предобученной моделью
                resized_arr = cv2.cvtColor(resized_arr, cv2.COLOR_GRAY2RGB)
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return data

def create_model():
    # Использование предобученной модели EfficientNet
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Размораживаем последние слои для fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    # Добавляем свои слои
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def plot_training_history(history):
    # Создаем списки для хранения значений
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    
    # Собираем значения из истории
    for epoch in range(len(history.history['accuracy'])):
        acc.append(float(history.history['accuracy'][epoch]))
        val_acc.append(float(history.history['val_accuracy'][epoch]))
        loss.append(float(history.history['loss'][epoch]))
        val_loss.append(float(history.history['val_loss'][epoch]))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def main():
    # Загрузка данных
    train = get_training_data('chest_xray/train')
    val = get_training_data('chest_xray/val')
    test = get_training_data('chest_xray/test')
    
    # Подготовка данных
    x_train = np.array([item[0] for item in train]) / 255.0
    y_train = np.array([item[1] for item in train])
    
    x_val = np.array([item[0] for item in val]) / 255.0
    y_val = np.array([item[1] for item in val])
    
    x_test = np.array([item[0] for item in test]) / 255.0
    y_test = np.array([item[1] for item in test])
    
    # Вычисление весов классов для борьбы с дисбалансом
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))
    
    # Аугментация данных
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
    val_generator = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE)
    
    # Создание модели
    model = create_model()
    
    # Компиляция модели с оптимизированными параметрами
    optimizer = Adam(learning_rate=0.0001)  # Уменьшенный learning rate
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Коллбэки
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            mode='max'
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Обучение модели
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=len(x_train) // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        # Оценка модели
        loss, accuracy = model.evaluate(x_test, y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        
        # Предсказания
        y_pred_proba = model.predict(x_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Визуализация результатов
        plot_training_history(history)
        plot_confusion_matrix(y_test, y_pred)
        plot_roc_curve(y_test, y_pred_proba)
        
        # Вывод метрик
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Pneumonia', 'Normal']))
        
        # Сохранение модели
        model.save('improved_pneumonia_model.h5')
        
    except Exception as e:
        print(f"Error during training: {e}")
        # Сохраняем модель даже в случае ошибки, если она была создана
        if 'model' in locals():
            model.save('improved_pneumonia_model.h5')

if __name__ == "__main__":
    main() 