from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = load_iris()
datasets = train_test_split(iris.data, iris.target,
                            test_size=0.3, random_state=285755)


train_data, test_data, train_labels, test_labels = datasets

scaler = StandardScaler()

# we fit the train data
scaler.fit(train_data)

# scaling the train data, Xscaled = (X - mu) / sigma
# przesuwanie danych do środka, mniejsze rozproszenie
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# creating an classifier from the model:
for layers in [(2,), (3,), (3,3,)]:
    mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=2000, learning_rate_init=0.01)

    # let's fit the training data to our model
    mlp.fit(train_data, train_labels)

    # predictions_train = mlp.predict(train_data)
    print(f"layers: {layers}")
    # print(accuracy_score(predictions_train, train_labels))
    predictions_test = mlp.predict(test_data)
    accuracy = accuracy_score(predictions_test, test_labels)
    print(accuracy_score(predictions_test, test_labels))
    print()

