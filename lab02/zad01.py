import pandas as pd
import numpy as np

missing_values = ["nan", "-"]
df = pd.read_csv("iris_with_errors.csv", na_values=missing_values)

# wyświetlanie statystyk brakującyhc danych
print(df.isnull().sum())

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        mean_value = df[col][(df[col] > 0) & (df[col] <= 15)].mean()
        df[col] = df[col].apply(lambda x: mean_value if x < 0 or x > 15 or np.isnan(x) else x)


correct_names = {"Setosa", "Virginica", "Versicolor"}

incorrect_species_indices = []
for i, species in enumerate(df.values[:, 4]):
    if species not in correct_names:
        incorrect_species_indices.append(i)

if incorrect_species_indices:
    for idx in incorrect_species_indices:
        corrected_species = df.iat[idx, 4].capitalize()
        if corrected_species == "Versicolour":
            corrected_species = "Versicolor"
        df.iat[idx, 4] = corrected_species

print(df)
