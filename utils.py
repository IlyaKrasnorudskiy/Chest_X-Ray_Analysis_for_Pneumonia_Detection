import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_class_distribution(train_data, labels=['Pneumonia', 'Normal']):
    """
    Визуализирует распределение классов в тренировочном наборе.
    
    Args:
        train_data (list): Тренировочные данные
        labels (list): Метки классов
    """
    l = [labels[i[1]] for i in train_data]
    sns.set_style('darkgrid')
    plt.figure(figsize=(6,4))
    sns.countplot(l)
    plt.title('Class distribution in training set')
    plt.show()

def plot_sample_images(train_data, labels=['PNEUMONIA', 'NORMAL'], num_samples=2):
    """
    Визуализирует примеры изображений из тренировочного набора.
    
    Args:
        train_data (list): Тренировочные данные
        labels (list): Метки классов
        num_samples (int): Количество примеров для каждого класса
    """
    plt.figure(figsize=(5*num_samples, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(train_data[i][0], cmap='gray')
        plt.title(labels[train_data[i][1]])
    plt.show()

def plot_training_history(history):
    """
    Визуализирует историю обучения модели.
    
    Args:
        history: История обучения модели
    """
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

def plot_confusion_matrix(y_true, y_pred):
    """
    Визуализирует матрицу ошибок.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """
    Визуализирует ROC-кривую.
    
    Args:
        y_true: Истинные метки
        y_pred_proba: Вероятности предсказаний
    """
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