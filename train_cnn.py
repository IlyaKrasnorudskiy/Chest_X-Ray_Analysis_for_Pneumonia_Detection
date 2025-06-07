import tensorflow as tf
import os
import numpy as np
from data_loader import get_training_data, prepare_data, create_data_generators
from model import create_model
from training import compile_model, train_model, evaluate_model
from utils import (plot_class_distribution, plot_sample_images, 
                  plot_training_history, plot_confusion_matrix, plot_roc_curve)

# Конфигурация
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.0001

def main():
    # Проверка GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"GPUs available: {[d.name for d in physical_devices]}")
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"Could not set memory growth: {e}")
    else:
        print("WARNING: GPU not found! Вычисления будут идти на CPU.")
    
    # Загрузка данных
    print("\nLoading data...")
    train_data = get_training_data('chest_xray/train', IMG_SIZE)
    val_data = get_training_data('chest_xray/val', IMG_SIZE)
    test_data = get_training_data('chest_xray/test', IMG_SIZE)
    
    # Визуализация данных
    print("\nVisualizing data distribution...")
    plot_class_distribution(train_data)
    plot_sample_images(train_data)
    
    # Подготовка данных
    print("\nPreparing data...")
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(
        train_data, val_data, test_data, IMG_SIZE
    )
    
    # Создание генераторов данных
    print("\nCreating data generators...")
    train_generator, val_generator = create_data_generators(
        x_train, y_train, x_val, y_val, BATCH_SIZE
    )
    
    # Создание и компиляция модели
    print("\nCreating and compiling model...")
    model = create_model(IMG_SIZE)
    model = compile_model(model, LEARNING_RATE)
    
    # Вычисление весов классов
    print("\nComputing class weights...")
    class_weights = {
        0: len(y_train) / (2 * np.sum(y_train == 0)),
        1: len(y_train) / (2 * np.sum(y_train == 1))
    }
    
    # Обучение модели
    print("\nTraining model...")
    history = train_model(
        model, train_generator, val_generator,
        x_train, x_val, y_val, class_weights,
        EPOCHS, BATCH_SIZE
    )
    
    # Визуализация истории обучения
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Оценка модели
    print("\nEvaluating model...")
    loss, accuracy, predictions = evaluate_model(model, x_test, y_test)
    
    # Визуализация результатов
    print("\nPlotting evaluation results...")
    plot_confusion_matrix(y_test, predictions)
    plot_roc_curve(y_test, model.predict(x_test))

if __name__ == "__main__":
    print(tf.__version__)
    main()
