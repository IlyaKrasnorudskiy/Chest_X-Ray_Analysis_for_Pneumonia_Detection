import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import gc
from sklearn.utils import class_weight

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32  # Увеличиваем размер батча для лучшей стабильности
EPOCHS = 30
LEARNING_RATE = 1e-4  # Возвращаемся к более высокой скорости обучения

# Вспомогательные функции

def получить_генераторы_данных(путь_к_датасету, размер_изображения, размер_пакета):
    # Calculate class weights
    train_dir = путь_к_датасету
    normal_count = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    pneumonia_count = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    total = normal_count + pneumonia_count
    class_weights = {
        0: total / (2 * normal_count),  # NORMAL
        1: total / (2 * pneumonia_count)  # PNEUMONIA
    }

    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',
        brightness_range=[0.9, 1.1]
    )

    # Only rescaling for validation and test
    val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        путь_к_датасету,
        target_size=(размер_изображения, размер_изображения),
        batch_size=размер_пакета,
        class_mode='binary',
        color_mode='rgb',
        shuffle=True
    )

    validation_generator = val_test_datagen.flow_from_directory(
        'chest_xray/val',
        target_size=(размер_изображения, размер_изображения),
        batch_size=размер_пакета,
        class_mode='binary',
        color_mode='rgb',
        shuffle=True
    )

    test_generator = val_test_datagen.flow_from_directory(
        'chest_xray/test',
        target_size=(размер_изображения, размер_изображения),
        batch_size=размер_пакета,
        class_mode='binary',
        color_mode='rgb',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator, class_weights

def построить_архитектуру_модели(размер_изображения):
    # Load pre-trained EfficientNetV2L
    base_model = EfficientNetV2L(
        include_top=False,
        weights='imagenet',
        input_shape=(размер_изображения, размер_изображения, 3)
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Create the model with attention mechanism
    inputs = layers.Input(shape=(размер_изображения, размер_изображения, 3))
    x = base_model(inputs)
    
    # Global context
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    # Dense layers with residual connections
    dense1 = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(0.3)(dense1)
    
    dense2 = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    dense2 = layers.Dropout(0.2)(dense2)
    
    # Residual connection
    residual = layers.Dense(256, activation='relu')(x)
    dense2 = layers.Add()([dense2, residual])
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(dense2)
    
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model with focal loss
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model, base_model

def обучение_модели(модель, генераторы, эпохи, оптимизатор):
    # Create logs directory for TensorBoard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]

    # First phase: Train only the top layers
    print("Phase 1: Training top layers...")
    history1 = model.fit(
        генераторы[0],
        validation_data=генераторы[1],
        epochs=эпохи,
        callbacks=callbacks,
        class_weight=генераторы[3]
    )

    # Second phase: Fine-tune the EfficientNet layers
    print("Phase 2: Fine-tuning EfficientNet layers...")
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 250  # Оптимальное количество замороженных слоев
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Очищаем память перед второй фазой
    tf.keras.backend.clear_session()
    gc.collect()

    history2 = model.fit(
        генераторы[0],
        validation_data=генераторы[1],
        epochs=эпохи,
        callbacks=callbacks,
        class_weight=генераторы[3]
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy, test_auc, test_precision, test_recall = model.evaluate(генераторы[2])
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    # Get predictions
    predictions = model.predict(генераторы[2])
    y_pred = (predictions > 0.5).astype(int)
    y_true = генераторы[2].classes

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['accuracy'], label='Phase 1 Train')
    plt.plot(history1.history['val_accuracy'], label='Phase 1 Val')
    plt.plot(history2.history['accuracy'], label='Phase 2 Train')
    plt.plot(history2.history['val_accuracy'], label='Phase 2 Val')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history1.history['loss'], label='Phase 1 Train')
    plt.plot(history1.history['val_loss'], label='Phase 1 Val')
    plt.plot(history2.history['loss'], label='Phase 2 Train')
    plt.plot(history2.history['val_loss'], label='Phase 2 Val')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# Основной запуск
if __name__ == "__main__":
    # Set memory growth for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Ограничиваем использование памяти GPU
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB
                )
        except RuntimeError as e:
            print(e)

    # Train the model
    train_generator, validation_generator, test_generator, class_weights = получить_генераторы_данных('chest_xray/train', IMG_SIZE, BATCH_SIZE)
    model, base_model = построить_архитектуру_модели(IMG_SIZE)
    обучение_модели(model, [train_generator, validation_generator, test_generator], EPOCHS, LEARNING_RATE) 