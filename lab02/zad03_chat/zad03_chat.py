from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

# Load the iris dataset
iris = pd.read_csv("../iris1.csv")

# Extract features and target variable
X = iris.iloc[:, :2]
y = iris['variety']

# Convert target variable to numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Original Data Statistics
print("Original Data Statistics:")
print(X.describe())

# Plot Original Data
plt.figure(figsize=(10, 6))
for i, variety in enumerate(iris['variety'].unique()):
    plt.scatter(X[y == variety].iloc[:, 0], X[y == variety].iloc[:, 1], label=variety)
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.savefig("original_data_plot.png")
plt.close()

# Z-score normalization
scaler_z = StandardScaler()
X_z = scaler_z.fit_transform(X)

# Z-score Normalized Data Statistics
print("\nZ-score Normalized Data Statistics:")
print(pd.DataFrame(X_z, columns=X.columns).describe())

# Plot Z-score Normalized Data
plt.figure(figsize=(10, 6))
for i, variety in enumerate(iris['variety'].unique()):
    plt.scatter(X_z[y == variety, 0], X_z[y == variety, 1], label=variety)
plt.title("Z-score Normalized Data")
plt.xlabel("Feature 1 (Z-score normalized)")
plt.ylabel("Feature 2 (Z-score normalized)")
plt.legend()
plt.savefig("z_score_normalized_plot.png")
plt.close()

# Min-max normalization
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)

# Min-max Normalized Data Statistics
print("\nMin-Max Normalized Data Statistics:")
print(pd.DataFrame(X_mm, columns=X.columns).describe())

# Plot Min-max Normalized Data
plt.figure(figsize=(10, 6))
for i, variety in enumerate(iris['variety'].unique()):
    plt.scatter(X_mm[y == variety, 0], X_mm[y == variety, 1], label=variety)
plt.title("Min-Max Normalized Data")
plt.xlabel("Feature 1 (Min-Max normalized)")
plt.ylabel("Feature 2 (Min-Max normalized)")
plt.legend()
plt.savefig("min_max_normalized_plot.png")
plt.close()

# prompty:
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import pandas as pd
# import matplotlib.pyplot as plt

# iris = pd.read_csv("../iris1.csv")

# X = iris.iloc[:, :2]
# y = iris['variety']

# There is this code. Create (save to different files) three scatter plots, where every point on the plot is a single iris. First plot should use original data.
# Second one should use z-score normalization. Third one shlud use min-max normalization. Also print statistics (mean, min, max, standard deviation) for every
# version of dataset.

# ValueError: 'c' argument must be a color, a sequence of colors, or a sequence of numbers, not 0 
# replace color bar with legend

