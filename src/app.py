import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

@st.cache_resource
def load_model_and_preprocessor():
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build paths to the artifacts
    preprocessor_path = os.path.join(current_dir, 'artifacts', 'preprocessor.pkl')
    model_path = os.path.join(current_dir, 'artifacts', 'kmeans_model.pkl')
    
    # Check if files exist and provide helpful error messages
    if not os.path.exists(preprocessor_path):
        st.error(f"Preprocessor file not found at: {preprocessor_path}")
        st.stop()
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    return preprocessor, model

preprocessor, model = load_model_and_preprocessor()

st.title("Customer Segmentation Prediction")

uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])
required_columns = ['annual_income', 'spending_score', 'savings', 'age']

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("First few rows of the dataset:")
    st.dataframe(df.head())

    if not all(col in df.columns for col in required_columns):
        st.error(f"Input CSV must contain the following columns: {', '.join(required_columns)}")
    else:
        if st.button("Predict"):
            # Preprocess and predict
            df_proc = df.copy()
            df_proc[required_columns] = preprocessor.transform(df_proc[required_columns])
            df_proc['cluster'] = model.predict(df_proc[required_columns])

            st.write("Predicted Clusters:")
            st.dataframe(df_proc[required_columns + ['cluster']])

            # Cluster characteristics
            st.write("Cluster Characteristics (mean values):")
            cluster_stats = df.groupby(df_proc['cluster'])[required_columns].mean().reset_index()
            cluster_stats.rename(columns={'cluster': 'Cluster'}, inplace=True)
            st.dataframe(cluster_stats)

            # Visualization
            df_proc['cluster'] = df_proc['cluster'].astype(str)  # Ensure clusters are categorical for color mapping
            fig = px.scatter(
                df_proc,
                x='annual_income',
                y='spending_score',
                color='cluster',
                title='Customer Segmentation Clusters',
                labels={'annual_income': 'Annual Income', 'spending_score': 'Spending Score'},
                color_discrete_sequence=px.colors.qualitative.Set1  # Use a vibrant, distinct palette
            )
            st.plotly_chart(fig)
else:
    st.write("Upload a CSV file to begin.")
