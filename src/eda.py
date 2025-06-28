import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

def perform_eda(df):
    """ Perform exploratory data analysis on the given DataFrame and save plots."""
    os.makedirs('artifacts', exist_ok=True)

    # Display basic information about the DataFrame
    print("DataFrame Info:")
    print(df.info())
    print("\nDataFrame Description:")
    print(df.describe())

    # Display the first few rows of the DataFrame
    print("\nFirst few rows of the DataFrame:")
    print(df.head())

    # correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.savefig('artifacts/correlation_matrix.png')
    plt.close()

    # Feature distributions
    for column in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.savefig(f'artifacts/distribution_{column}.png')
        plt.close()

    # Elbow method for optimal number of clusters
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    inertia = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(pca_data)
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(range(1, 11))
    plt.savefig('artifacts/elbow_method.png')
    plt.close()
    print("EDA completed and plots saved in 'artifacts' directory.")

if __name__ == "__main__":
    # Example usage
    # Load a sample dataset
    df = pd.read_csv(r'C:\Users\SEGUN\customer segmentation\data\week5_segmentation_data.csv')
    perform_eda(df)