import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('/content/Mall_Customers.csv')

# Optional: drop irrelevant columns (e.g., customer ID)
df = df.drop(['customerID'], axis=1, errors='ignore')

# Handle categorical columns
df = pd.get_dummies(df)

# Fill missing values
df.fillna(0, inplace=True)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# -------------------------------
# Elbow Method to Find Optimal K
# -------------------------------
wcss = []
k_values = range(1, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_data)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.xticks(k_values)
plt.show()

# -------------------------------
# KMeans Clustering (after deciding k=3 from Elbow)
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
df['Cluster'] = clusters

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Transform centroids to PCA space
centroids_pca = pca.transform(kmeans.cluster_centers_)

# Plotting with centroids
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette='Set1', s=60, alpha=0.6)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Customer Segments Visualized via PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()
