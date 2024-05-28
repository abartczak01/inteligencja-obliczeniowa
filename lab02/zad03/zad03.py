from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("../iris1.csv")

X = iris.iloc[:, :2]
y = iris['variety']

def add_labels_and_legend():
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend()

def plot_scaled_dataset(scaler, scaler_name):
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    print("Statistics for", scaler_name)
    print(X_scaled_df.describe(), '\n')
    plt.figure(figsize=(8, 6))
    for species in iris['variety'].unique():
        subset_scaled = X_scaled[y == species]
        plt.scatter(subset_scaled[:, 0], subset_scaled[:, 1], label=species)

    plt.title(f'{scaler_name} Scaled Dataset')
    add_labels_and_legend()
    plt.savefig(f"zad03_{scaler_name.lower().replace('-', '_')}.png")


def plot_original():
    print("Statistic for oirignal data")
    print(X.describe(), '\n')
    plt.figure(figsize=(8, 6))
    for species in iris['variety'].unique():
        subset = iris[iris['variety'] == species]
        plt.scatter(subset.iloc[:, 0], subset.iloc[:, 1], label=species)

    plt.title('Original Dataset')
    add_labels_and_legend()
    plt.savefig("zad03_original.png")

plot_original()

# przeskalowanie do rozkładu normalnego z mu=0, sigma=1
plot_scaled_dataset(StandardScaler(), 'Z-Score')

# wszystkie wartości są z zakresu 0-1
plot_scaled_dataset(MinMaxScaler(), 'Min-Max')
