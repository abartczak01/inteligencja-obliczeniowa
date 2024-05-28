import os
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.layers import Dropout
from keras.applications.resnet50 import preprocess_input, decode_predictions

# Ścieżka do katalogu z danymi treningowymi i testowymi
train_data_dir = '../../lab05/zad03/dataset_dogs_vs_cats/train/'
test_data_dir = '../../lab05/zad03/dataset_dogs_vs_cats/test/'

# Parametry treningowe
batch_size = 32
epochs = 10
num_classes = 2  # Koty i psy

# Generatory danych
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical')

print("po train generator")
print(train_generator.samples)

validation_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical')

print("po validation generator")

# Załaduj istniejący model ResNet50 bez warstw końcowych
base_model = ResNet50(weights='imagenet', include_top=False)

# Dodaj nowe warstwy do modelu
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Nowy model łączący istniejący model z nowymi warstwami
model = Model(inputs=base_model.input, outputs=predictions)

print("po dodaniu nowych warstw")

# Zamrożenie wszystkich warstw w bazowym modelu ResNet50
for layer in base_model.layers:
    layer.trainable = False

# Kompilacja modelu
model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

print("po skompilowaniu")

# Trenowanie modelu
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size - 1,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size - 1
    )

print("po trenowaniu")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Wykresy strat
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(epochs), loss, 'b', label='Training Loss')
plt.plot(range(epochs), val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Wykresy dokładności
plt.subplot(1, 2, 2)
plt.plot(range(epochs), acc, 'b', label='Training Accuracy')
plt.plot(range(epochs), val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('learning_curve.png')

# Oceń dokładność modelu na danych testowych
score =  model.evaluate(validation_generator, steps=validation_generator.samples // batch_size - 1)
# score =  model.evaluate(validation_generator)
print("Test Accuracy:", score[1])

# Zapisz model
model.save('trained_model.keras')

# 195/195 ━━━━━━━━━━━━━━━━━━━━ 220s 1s/step - accuracy: 0.4899 - loss: 0.6932
# Test Accuracy: 0.4932692348957062