import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set_theme()

df = pd.read_csv("../data/iris1.csv")

all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = df['variety'].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=285755)


# k najbliżisych sąsiadów
# number = 3
for number in [3, 5, 11]:
    knn = KNeighborsClassifier(n_neighbors=number, metric='euclidean')
    knn.fit(train_inputs, train_classes)
    prediction = knn.predict(test_inputs)
    cm = confusion_matrix(test_classes, prediction)
    accuracy = accuracy_score(prediction, test_classes)
    print("--------------------------------------")
    print(f"k-najbliższych sąsiadów, n = {number}")
    print(f"macierz błędu:\n{cm}")
    print(f"dokładność: {round(accuracy * 100, 2)}%")

model = GaussianNB()
model.fit(train_inputs, train_classes)
prediction = model.predict(test_inputs)
cm = confusion_matrix(test_classes, prediction)
accuracy = accuracy_score(prediction, test_classes)
print("--------------------------------------")
print(f"Naive Bayes")
print(f"macierz błędu:\n{cm}")
print(f"dokładność: {round(accuracy * 100, 2)}%")

# najlepszy są knn3, knn11, naive bayes