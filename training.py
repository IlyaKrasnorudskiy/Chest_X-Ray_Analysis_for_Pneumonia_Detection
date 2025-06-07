import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import numpy as np

def compile_model(model, learning_rate=0.0001):
    """
    Компилирует модель с заданными параметрами.
    
    Args:
        model: Модель для компиляции
        learning_rate (float): Скорость обучения
        
    Returns:
        tensorflow.keras.models.Model: Скомпилированная модель
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_callbacks():
    """
    Создает коллбэки для обучения модели.
    
    Returns:
        list: Список коллбэков
    """
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

def train_model(model, train_generator, val_generator, x_train, x_val, y_val, 
                class_weights, epochs=12, batch_size=32):
    """
    Обучает модель.
    
    Args:
        model: Модель для обучения
        train_generator: Генератор тренировочных данных
        val_generator: Генератор валидационных данных
        x_train: Тренировочные данные
        x_val: Валидационные данные
        y_val: Валидационные метки
        class_weights: Веса классов
        epochs (int): Количество эпох
        batch_size (int): Размер батча
        
    Returns:
        tensorflow.keras.callbacks.History: История обучения
    """
    history = model.fit(
        train_generator,
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=get_callbacks(),
        class_weight=class_weights
    )
    return history

def evaluate_model(model, x_test, y_test):
    """
    Оценивает модель на тестовых данных.
    
    Args:
        model: Обученная модель
        x_test: Тестовые данные
        y_test: Тестовые метки
        
    Returns:
        tuple: (loss, accuracy, predictions)
    """
    # Оценка модели
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Предсказания
    predictions = (model.predict(x_test) > 0.5).astype("int32").reshape(-1)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Pneumonia (Class 0)', 'Normal (Class 1)']))
    
    return loss, accuracy, predictions 