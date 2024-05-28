from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("iris1.csv")
X = iris.drop('variety', axis=1)
y = iris['variety']

explained_variance_ratio = 0.0
n_components = 0

while explained_variance_ratio < 0.95:
    n_components += 1
    pca_iris = PCA(n_components=n_components)
    pca_iris.fit(X)
    explained_variance_ratio = pca_iris.explained_variance_ratio_.sum()

print("number of components:", n_components)

X_pca = pca_iris.transform(X)

df_pca = pd.DataFrame(data=X_pca, columns=[f'PC{i}' for i in range(1, n_components + 1)])
df_pca = pd.concat([df_pca, y], axis=1)

plt.figure(figsize=(8, 6))
for species in df_pca['variety'].unique():
    plt.scatter(df_pca.loc[df_pca['variety'] == species, 'PC1'],
                df_pca.loc[df_pca['variety'] == species, 'PC2'],
                label=species)

plt.title('PCA of Iris Dataset')
plt.legend()
plt.savefig("zad02.png")
