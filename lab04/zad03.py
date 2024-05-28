import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../data/diabetes1.csv")

all_classes = df.iloc[:, -1].values
all_inputs = df.iloc[:, :-1].values

print(all_inputs)
label_encoder = LabelEncoder()
all_classes = label_encoder.fit_transform(all_classes)

(train_data, test_data, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=285755)

def classification(layers, activation):
    mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=500, activation=activation)
    mlp.fit(train_data, train_classes)
    predictions = mlp.predict(test_data)
    accuracy = accuracy_score(predictions, test_classes)
    cm = confusion_matrix(test_classes, predictions)
    print(f"layers: {layers}, acitvation: {activation}")
    print("confusion matrix:")
    print(cm)
    print(f"accuracy: {accuracy}")

    false_positive = cm[0][1]
    false_negative = cm[1][0]

    print(f"False Positive: {false_positive}")
    print(f"False Negative: {false_negative}")
    return accuracy


activation_scores = {}
layers_scores = {}
for activation in ['relu', 'identity', 'logistic', 'tanh']:
    activation_scores[activation] = 0
    for hidden_layer_sizes in [(6, 3), (10, 10, 10, 10, 10), 50]:
        accuracy = classification(hidden_layer_sizes, activation)
        activation_scores[activation] += accuracy
        if hidden_layer_sizes not in layers_scores:
            layers_scores[hidden_layer_sizes]=0
        layers_scores[hidden_layer_sizes] += accuracy
        print('-'*50, '\n')

print(activation_scores)
print(layers_scores)
best_activation = max(activation_scores, key=activation_scores.get)
print("best activation:", best_activation)
best_layers = max(layers_scores, key=layers_scores.get)
print("best layers:", best_layers)

# dla danych warstw [(6, 3), (10, 10, 10, 10, 10), 50] zwykle wygrywa 50,
# dla funkcji aktywacji najczęściej relu, trochę rzadziej identity

# false negative są gorsze, ponieważ chora osoba zostaje zdiagnozowana jako zdrowa i nie podejmuje leczenia
# false positive powodują tylko dodatkowy stres i przejście do kolejnych badań pacjenta