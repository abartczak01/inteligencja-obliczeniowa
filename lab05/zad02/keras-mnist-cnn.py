import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History, ModelCheckpoint

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1)  # Save original labels for confusion matrix

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# checkpoint to zapisywania najlepszego modelu
checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', save_best_only=True)

# Train model
history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[history, checkpoint])

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('predicted.png')

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.savefig('loss.png')

# Display 25 images from the test set with their predicted labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.savefig('images.png')

# a) reshape zmienia kształt danych z (liczba_obrazów, 28, 28) na (liczba_obrazów, 28, 28, 1), gdzie 1 oznacza liczbę kanałów (1 - czarno-białe)
# to_categorical zamienia etykiety (nawzy klas) na forme one-hot encoding
# np.argmax dla danych wejściowych w postaci one-hot encoding, np.argmax zwraca oryginalne etykiety klas jako jednowymiarową tablicę; tutaj słży do zachowania oryginalnych etykiet tekstowych

# b) 1. warstwa wejściowa Conv2D 
# - na wejściu: czarno-białe obrazy o wymiarach 28x28, 
# - na wyjściu mapy cech uzyskane dzięki filtrom
# 2. Warstwa Pooling (MaxPooling2D)
# - Na wejściu: Mapy cech uzyskane z poprzedniej warstwy.
# - Na wyjściu: Zredukowane mapy cech, które są uzyskiwane poprzez zastosowanie operacji max-pooling na mapach cech wejściowych
# 3. Warstwa Flatten:
# - na wejściu: Mapy cech wyjściowe z warstwy MaxPooling2D. 
# - na wyjściu: jednowymiarowy wektor danych, który jest w pełni spłaszczonym zestawem danych (z wielowymiarowej macierzy na jednowymiarowy wektor)
# 4. Warstwa Dense (64 neuronów)
# - na wejściu: jednowymiarowy wektor z warstwy flatten
# - na wyjściu: przekształcone wartości (suma elementów: wartość z neurona z porpzedniej warswty przemnożonoa przez odpowiednią wagę
#; suma zaktywowana funkcją relu)
# 5. Warstwa Dense (10 neuronów), warstwa wyjściowa
# - na wejściu: dane wyjściowe z poprzedniej warstwy Dense (64 neuronów).
# - na wyjściu:  Ostateczne wyniki klasyfikacji. Warstwa ta zawiera 10 neuronów, które reprezentują 10 możliwych klas (cyfry od 0 do 9). 
# Aktywacja softmax jest stosowana w tej warstwie, aby uzyskać prawdopodobieństwa przynależności do każdej z klas.

# c) najwięcej błędów pojawia się przy przewidywaniu cyfry 5; 21 razy pomylono piątkę z trójką, a 19 razy z szóstką
# d) widoczny jest przypadek przeuczenia, na wykresie z loss widać że train loss spada, validation loss zaczyna w pewnym momencie rosnąć