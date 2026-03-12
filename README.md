# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the Mall Customers dataset.
2. Select important features such as Age, Annual Income, and Spending Score.
3. Standardize the data using StandardScaler.
4. Apply K-Means clustering and determine the number of clusters using the Elbow Method.
5. Visualize the clusters and evaluate the result using Silhouette Score.


## Program:
```
# Customer Segmentation using K-Means Clustering

import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Display first few rows
print(data.head())

# Display column names
print("\nColumns in dataset:")
print(data.columns)

print("\nName: POOJA A")
print("Register Number: 212225040300")

# Select features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal clusters
inertia = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow graph
plt.figure(figsize=(8,5))
plt.plot(range(1,11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# Apply K-Means with optimal clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Calculate Silhouette Score
score = silhouette_score(X_scaled, data['Cluster'])
print("\nSilhouette Score:", score)

# Visualize clusters
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=data,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='viridis',
    s=100
)

# Plot cluster centroids
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:,1], centers[:,2], c='red', s=200, marker='X', label='Centroids')

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

```

## Output:
<img width="690" height="281" alt="image" src="https://github.com/user-attachments/assets/4602680e-ae19-4baa-b511-764f83430dc2" />
<img width="942" height="566" alt="image" src="https://github.com/user-attachments/assets/afa52f1a-5b53-4083-8257-30df2de81dce" />
<img width="1040" height="661" alt="image" src="https://github.com/user-attachments/assets/4088226d-85f5-4fb5-8a04-3adbde75c879" />




## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
