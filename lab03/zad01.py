import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/iris1.csv")

(train_set, test_set) = train_test_split(df, train_size=0.7, random_state=285755)

def classify_iris(sl, sw, pl, pw):
    if pw < 1:
        return ("Setosa")
    elif pw >= 1 and pw < 1.8:
        return ("Versicolor")
    else:
        return ("Virginica")

good_predictions = 0
total_instances = test_set.shape[0]

for i in range(total_instances):
    predicted_class = classify_iris(test_set.iloc[i, 0], test_set.iloc[i, 1], 
                                    test_set.iloc[i, 2], test_set.iloc[i, 3])

    if predicted_class == test_set.iloc[i, 4]:
        good_predictions += 1

accuracy = (good_predictions / total_instances) * 100
print("Liczba poprawnych predykcji:", good_predictions)
print("Dokładność modelu:", accuracy, "%")