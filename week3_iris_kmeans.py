import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target                         # Used only after clustering for comparison
Z = StandardScaler().fit_transform(X)   # Scale features for distance calculation

inertia, scores = [], []
for k in range(2, 8):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(Z)
    inertia.append(model.inertia_)
    scores.append(silhouette_score(Z, labels))

plt.plot(range(2, 8), inertia, marker="o")
plt.title("Elbow Method"); plt.xlabel("K"); plt.ylabel("Inertia"); plt.show()
plt.plot(range(2, 8), scores, marker="o")
plt.title("Silhouette Scores"); plt.xlabel("K"); plt.ylabel("Score"); plt.show()

model = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = model.fit_predict(Z)
print("Silhouette Score:", silhouette_score(Z, clusters))
print("Adjusted Rand Index:", adjusted_rand_score(y, clusters))

# PCA is only for plotting; model used all four features
points = PCA(n_components=2).fit_transform(Z)
plt.scatter(points[:, 0], points[:, 1], c=clusters, s=35)
plt.title("K-Means Clusters (PCA)"); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.show()

result = X.copy()
result["cluster"] = clusters + 1
print(result.groupby("cluster").mean())
print(result["cluster"].value_counts().sort_index())
