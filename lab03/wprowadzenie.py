import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("../data/iris1.csv")

# print(df)
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=13)

# print(test_set)
# print(test_set.shape[0])

trains_inputs = train_set[:, 0:4]
trains_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

# print(trains_inputs)
# print(trains_classes)

# print(test_inputs)
# print(test_classes)