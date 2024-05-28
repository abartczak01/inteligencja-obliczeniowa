import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("../data/iris1.csv")

all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = df['variety'].values

# sns.pairplot(df, hue='variety')
# plt.savefig('zad02.png')

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=285755)

dtc = DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)

accuracy = dtc.score(test_inputs, test_classes)
print(f"dokładność klasyifkatora: {round(accuracy * 100, 2)}%")

# wizualizacja drzewa decyzyjnego
plt.figure(figsize=(12, 8))
plot_tree(dtc, filled=True, feature_names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'], class_names=df['variety'].unique())
plt.savefig('tree.png')

predictions = dtc.predict(test_inputs)

# macierz błędów
cm = confusion_matrix(test_classes, predictions)

print("Macierz błędów:")
print(cm)

# zliczanie dobrych i złych dopasowań
wrong_ans = 0
correct_ans = 0

for i, row in enumerate(cm):
    for j, col in enumerate(row):
        if i == j:
            correct_ans += col
        else:
            wrong_ans += col

print("Liczba błędnych odpowiedzi:", wrong_ans)
print("Liczba poprawnych odpowiedzi:", correct_ans)
