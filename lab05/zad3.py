import sys
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
FAST_RUN = True
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3


def create_model(optimizer='adam', activation='relu', dropout=False):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same',
                     input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(MaxPooling2D((2, 2)))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    if dropout:
        model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation=activation, kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    if optimizer == 'adam':
        opt = Adam(learning_rate=0.001)
    else:
        opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate_model(model, train_data, test_data, model_name):
    # create data generator
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # prepare iterators
    train_it = datagen.flow_from_directory(train_data,
                                           class_mode='binary', batch_size=32, target_size=IMAGE_SIZE)

    test_it = datagen.flow_from_directory(test_data,
                                          class_mode='binary', batch_size=32, target_size=IMAGE_SIZE)
    # fit model
    history = model.fit(train_it,
                        validation_data=test_it,
                        epochs=5, verbose=1)

    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=1)
    print('> %.3f' % (acc * 100.0))

    summarize_diagnostics(history, model_name, test_it, model)


def summarize_diagnostics(history, model_name, test_it, model):
    # plot loss and accuracy
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    # plot loss
    axs[0].plot(history.history['loss'], color='blue', label='train')
    axs[0].plot(history.history['val_loss'], color='orange', label='test')
    axs[0].set_title('Cross Entropy Loss')
    axs[0].legend()

    # plot accuracy
    axs[1].plot(history.history['accuracy'], color='blue', label='train')
    axs[1].plot(history.history['val_accuracy'], color='orange', label='test')
    axs[1].set_title('Classification Accuracy')
    axs[1].legend()

    # save plot to file
    plt.savefig(f'{model_name}_learning_curves.png')
    plt.close()

    y_true = test_it.classes
    y_pred = model.predict(test_it, steps=len(test_it), verbose=1)
    y_pred_classes = (y_pred > 0.5).astype("int32")

    cm = confusion_matrix(y_true, y_pred_classes)
    print(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

    cr = classification_report(y_true, y_pred_classes, output_dict=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(pd.DataFrame(cr).iloc[:-1, :].T, annot=True, cmap='Blues')
    plt.title('Classification Report')
    plt.savefig(f'{model_name}_classification_report.png')
    plt.close()


# model1v2 = create_model(optimizer='adam', activation='relu', dropout=False)
# train_and_evaluate_model(model1v2, 'dataset_dogs_vs_cats/train/', 'dataset_dogs_vs_cats/test/', 'model1v2')

#74.615
model2 = create_model(optimizer='sgd', activation='relu', dropout=False)
train_and_evaluate_model(model2, 'dataset_dogs_vs_cats/train/', 'dataset_dogs_vs_cats/test/', 'model2')

#74.123
model3 = create_model(optimizer='adam', activation='relu', dropout=True)
train_and_evaluate_model(model3, 'dataset_dogs_vs_cats/train/', 'dataset_dogs_vs_cats/test/', 'model3')

#50.643
model4 = create_model(optimizer='adam', activation='sigmoid', dropout=False)
train_and_evaluate_model(model4, 'dataset_dogs_vs_cats/train/', 'dataset_dogs_vs_cats/test/', 'model4')
