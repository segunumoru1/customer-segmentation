import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pipeline import create_preprocessing_pipeline, preprocess_data
import mlflow
import mlflow.sklearn
import os
from sklearn.metrics import silhouette_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# set random seed for reproducibility
np.random.seed(42)

def train_and_evaluate():
    """Train a KMeans clustering model and evaluate its performance."""
    # Load the dataset
    df = pd.read_csv(r'C:\Users\SEGUN\customer segmentation\data\week5_segmentation_data.csv')

    # Create preprocessing pipeline
    preprocessor, features = create_preprocessing_pipeline()

    # Preprocess the data
    df, preprocessor, features = preprocess_data(df, preprocessor, features)

    # Initialize MLflow
    mlflow.set_experiment("customer_segmentation")

    with mlflow.start_run():
        # Train KMeans model
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(df[features])

        # Predict clusters
        df['cluster'] = kmeans.predict(df[features])

        # Calculate silhouette score
        silhouette_avg = silhouette_score(df[features], df['cluster'])
        print(f'Silhouette Score: {silhouette_avg}')

        # Log parameters and metrics to MLflow
        mlflow.log_param("n_clusters", 4)
        mlflow.log_metric("silhouette_score", float(silhouette_avg))

        # Log the model
        mlflow.sklearn.log_model(kmeans, "kmeans_model")
        mlflow.sklearn.log_model(preprocessor, "preprocessor")

        # Save the preprocessor and model locally
        os.makedirs('artifacts', exist_ok=True)
        joblib.dump(preprocessor, 'artifacts/preprocessor.pkl')
        joblib.dump(kmeans, 'artifacts/kmeans_model.pkl')

        # Visualize clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df['cluster'], palette='viridis')
        plt.title('KMeans Clustering Results')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.savefig('artifacts/kmeans_clustering.png')
        plt.close()

    print("Training and evaluation completed. Results saved in 'artifacts' directory.")
if __name__ == "__main__":
    train_and_evaluate()