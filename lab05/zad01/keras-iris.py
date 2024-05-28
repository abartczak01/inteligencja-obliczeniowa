import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tensorflow.keras.optimizers import SGD


# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the labels
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Define the model
# relu(x) = max(x, 0)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')
]) # accuracy 100%

# sigmoid(x) = 1 / (1 + exp(-x))
# model = Sequential([
#     Dense(64, activation='sigmoid', input_shape=(X_train.shape[1],)),
#     Dense(64, activation='sigmoid'),
#     Dense(y_encoded.shape[1], activation='softmax')
# ]) # accuracy 95.56%

# tanh(x) = sinh(x) / cosh(x), i.e. tanh(x) = ((exp(x) - exp(-x)) / (exp(x) + exp(-x)))
# model = Sequential([
#     Dense(64, activation='tanh', input_shape=(X_train.shape[1],)),
#     Dense(64, activation='tanh'),
#     Dense(y_encoded.shape[1], activation='softmax')
# ]) # accuracy 100.00%

# model = Sequential([
#     Dense(64, activation='softsign', input_shape=(X_train.shape[1],)),
#     Dense(64, activation='softsign'),
#     Dense(y_encoded.shape[1], activation='softmax')
# ]) # accuracy 100.00%

# Compile the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # accuracy 100%
# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) # accuracy 86.67%
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']) accuracy 100.00%
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['recall']) recall 100%, true positive rate, mierzy zdolność modelu do poprawnego zidentyfikowania wszystkich pozytywnych przypadków spośród wszystkich rzeczywistych pozytywnych przypadków w zbiorze danych.
# categorical_crossentropy jest używana, gdy model musi przewidywać prawdopodobieństwa przynależności do różnych klas
# mean_squared_error MSE dla problemów, takich jak przewidywanie cen, prognozowanie, regresja liniowa

# można dostosować szybkość uczenia sie modelu
# optimizer = SGD(learning_rate=0.01)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


y_train_dense = y_train.toarray()

# Train the model
history = model.fit(X_train, y_train_dense, epochs=100, validation_split=0.2)
# history = model.fit(X_train, y_train_dense, epochs=100, validation_split=0.2, batch_size=16)

# znajdowanie epoch z najlepszą wydajnością
best_epoch = np.argmax(history.history['val_accuracy']) + 1
print(f"Najlepsza epoka: {best_epoch}")


y_test_dense = y_test.toarray()
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test_dense, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.savefig('learning_curve.png')

# Save the model
model.save('iris_model.keras') # zmieniono rozszerzenie z h5 na keras

# Plot and save the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print(X_train.shape[1])
print(y_encoded.shape[1])

# a) StandardScaler służy do skalowania danych numerycznych tak, aby pasowały do standardowego rozkładu normalnego (czyli średnia, mu = 0; odchylenie, sigma = 1)
# b) OneHotEnocder dla każdej klasy tworzy kolumnę (kolumna to wektor zer i jednej jedynki, gdzie 1 oznacza przynależność do klasy). 
# c) Warstwa wejściowa ma 4 neurony - zależy to od liczby cech. X_train.shape[1] to właśnie liczba cech (z danych treningowych)
# Warstwa wyjściowa ma 3 neurony - zależy to od liczby klas. y_encoded.shape[1] to liczba unikalnych klas
# d) pod względem accuracy tanh, relu i softsign wypadły tak samo na 100%
# e) rożne optymalizatory dają różne wyniki. Można dostosować szybkość uczenia się w optymalizatorze.
# f) mozna zmienić rozmiar partii przez użycie parametru batch_size. Dla wiekszych partii wzrosła dokładność (accuracy). 
# dla wiekszych partii wartość loss jest mniejsza, przy 16 wydaje sie być bardziej stabliny (nie skacze na końcu)
# g) batch_size domyślnie w keras ma wartość 32, wyszło że najlepsza epoch to 37, co sie zgadza z wykresem, tam funckja po raz pierwszy osiągnęła najwyższą wydajność 
# krzywa sugeruje całkiem dobrze dobrany model. Na pierwszym wykresie widać, że większa liczba epoch nie dawała lepszego accuracy,
# a na drugim wykresie z widać wzrost wartości loss co sugeruje przetrenowanie modelu. Mniejsza liczba epoch mogłaby się okazać lepsza.
# h) Podsumowanie kodu: 
# 1. Pobrano dataset z irysami. 
# 2. Oddzielono cechy od klas. 
# 3. Przeskalowano cechy za pomocą StandardScaler, czyli w taki sposób, żeby wartości tych cech należały do standardowego układu normalnego. 
# 4. Dokonano kodowania one-hot na klasach z datasetu
# 5. Podzielono dataset na dane treningowe 70% i dane testowe 30%
# 6. Pobrano wytrenowany model z pliku iris_model.h5
# 7. Douczono model o kolejne 10 epoch używając wcześniej wydzielonych danych
# 8. Zapisano douczony model w pliu updated_iris_model.h5
# 9. Obliczono accuracy i loss dla nowego modelu, używając wcześniej wyznaczonego testowego datasetu