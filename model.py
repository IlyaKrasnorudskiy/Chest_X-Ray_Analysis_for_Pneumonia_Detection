from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization

def create_model(img_size=150):
    """
    Создает модель CNN для классификации рентгеновских снимков.
    
    Args:
        img_size (int): Размер входного изображения
        
    Returns:
        tensorflow.keras.models.Sequential: Скомпилированная модель
    """
    model = Sequential([
        # Первый сверточный блок
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(img_size, img_size, 1)),
        BatchNormalization(),
        MaxPool2D((2,2), padding='same'),

        # Второй сверточный блок
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Dropout(0.1),
        BatchNormalization(),
        MaxPool2D((2,2), padding='same'),

        # Третий сверточный блок
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2,2), padding='same'),

        # Четвертый сверточный блок
        Conv2D(128, (3,3), activation='relu', padding='same'),
        Dropout(0.2),
        BatchNormalization(),
        MaxPool2D((2,2), padding='same'),

        # Пятый сверточный блок
        Conv2D(256, (3,3), activation='relu', padding='same'),
        Dropout(0.2),
        BatchNormalization(),
        MaxPool2D((2,2), padding='same'),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    return model 