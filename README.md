# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Dataset.
2.Import the dataset to start the clustering analysis process.
3.Explore the Data
4.Analyze the dataset to understand distributions, patterns, and key characteristics.
5.Select Relevant Features
6.Identify the most informative features to improve clustering accuracy and relevance.
7.Preprocess the Data
8.Clean and scale the data to prepare it for clustering.
9.Determine Optimal Number of Clusters
10.Use techniques like the elbow method to find the ideal number of clusters.
11.Train the Model with K-Means Clustering
12.Apply the K-Means algorithm to group data points into clusters based on similarity.
13.Analyze and Visualize Clusters
14.Examine and visualize the resulting clusters to interpret patterns and relationships.

## Program:
```
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: Janani Saraswathi.S
RegisterNumber:212225230110
import os
os.environ["OMP_NUM_THREADS"] = "1" 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")

data = pd.read_csv("CustomerData.csv")
print(data.head())
print(data.columns)
print("\nName: Janani Saraswathi S")
print("Reg. No: 212225230110")
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
kmeans.fit(X_scaled)
data['Cluster'] = kmeans.labels_
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=data,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='viridis',
    s=100,
    alpha=0.7
)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 1], centers[:, 2], c='red', s=200, alpha=0.75, marker='X', label='Centroids')

plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
*/
```

## Output:
<img width="880" height="744" alt="Screenshot 2026-03-11 132406" src="https://github.com/user-attachments/assets/77776382-906b-4abc-bf26-c84103eccb53" />
<img width="1115" height="613" alt="Screenshot 2026-03-11 132525" src="https://github.com/user-attachments/assets/f972d55a-d8f5-4dc2-ba12-fc9d45d1ee77" />



## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
