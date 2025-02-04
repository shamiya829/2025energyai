import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import warnings
import os

warnings.filterwarnings('ignore')

# Streamlit layout
st.title('Energy A.I. Hackathon 2025 Workflow - Fractimus')
st.write('Authors: Isaac Xu, Shamiya Lin, Christopher Chen, Alex Huynh')
st.write('Petroleum and Geosystems Engineering, School of Information')

# File Name Input
st.header('Enter the File Name')
file_name = st.text_input("Enter the name of the file (with extension, e.g., 'data.csv')", "")

if file_name:
    # Check if the file exists in the current directory
    if os.path.exists(file_name):
        # Load the dataset
        if file_name.endswith('.csv'):
            data = pd.read_csv(file_name)
        elif file_name.endswith('.xlsx'):
            data = pd.read_excel(file_name)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
        
        # Show first few rows of the dataset
        st.write(data.head())

        # Data Analysis section
        st.header('Data Analysis')
        st.subheader('Data Summary')
        st.write(data.describe())

        st.subheader('Correlation Heatmap')
        correlation = data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Feature Engineering section
        st.header('Feature Engineering')

        # Handle missing data (example: fill with median)
        st.subheader('Handling Missing Data')
        data.fillna(data.median(), inplace=True)
        st.write('Missing values have been imputed with the median.')

        # Feature Selection
        st.subheader('Feature Selection')
        st.write('Selecting relevant features based on domain knowledge and correlation.')

        # Let's assume some features and target column are known (replace these with actual column names)
        X = data.drop(columns=['Grid', 'Energy_Usage_Diesel', 'Energy_Usage_CNG'])
        y = data[['Energy_Usage_Grid', 'Energy_Usage_Diesel', 'Energy_Usage_CNG']]

        # Label Encoding for categorical features if any
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            label_encoders[col] = LabelEncoder()
            X[col] = label_encoders[col].fit_transform(X[col])

        # Data Split into Training and Testing sets
        st.subheader('Data Splitting')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f'Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}')

        # Model Training and Evaluation (RandomForestRegressor)
        st.header('Model Training: Random Forest')
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions on the test set
        st.subheader('Model Predictions')
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        st.subheader('Model Evaluation')
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f'Mean Squared Error: {mse:.2f}')
        st.write(f'R-Squared: {r2:.2f}')
        
        # Show predictions vs actual values
        st.subheader('Predictions vs Actual Values')
        prediction_df = pd.DataFrame({
            'Actual': y_test.values.flatten(),
            'Predicted': y_pred.flatten()
        })
        st.write(prediction_df.head())

        # Plot predictions vs actual
        fig, ax = plt.subplots()
        ax.scatter(y_test.values.flatten(), y_pred.flatten())
        ax.plot([min(y_test.values.flatten()), max(y_test.values.flatten())], 
                [min(y_test.values.flatten()), max(y_test.values.flatten())], color='red', linestyle='--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted')
        st.pyplot(fig)

        # Uncertainty Quantification (Bootstrap)
        st.header('Uncertainty Quantification: Bootstrap')
        
        # Define a function to perform bootstrap resampling and calculate uncertainty
        def bootstrap_uncertainty(model, X_train, y_train, n_iterations=1000):
            predictions = []
            for _ in range(n_iterations):
                # Sample with replacement
                X_resample, y_resample = resample(X_train, y_train)
                model.fit(X_resample, y_resample)
                pred = model.predict(X_test)
                predictions.append(pred)
            predictions = np.array(predictions)
            return predictions
        
        from sklearn.utils import resample
        
        st.subheader('Bootstrap Resampling')
        boot_predictions = bootstrap_uncertainty(model, X_train, y_train)
        
        # Calculate 95% confidence intervals for predictions
        lower_bound = np.percentile(boot_predictions, 2.5, axis=0)
        upper_bound = np.percentile(boot_predictions, 97.5, axis=0)
        
        st.write('95% Confidence Intervals for Predictions:')
        st.write(f'Lower Bound: {lower_bound}')
        st.write(f'Upper Bound: {upper_bound}')
        
        # Visualizing Uncertainty
        st.subheader('Prediction Interval Plot')
        fig, ax = plt.subplots()
        ax.plot(y_pred.flatten(), label='Predicted', color='blue')
        ax.fill_between(range(len(y_pred.flatten())), lower_bound, upper_bound, color='gray', alpha=0.3, label='95% CI')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Predicted Energy Usage')
        ax.legend()
        st.pyplot(fig)

        st.write("This concludes the workflow for predicting energy consumption with uncertainty quantification.")

    else:
        st.error(f"File '{file_name}' does not exist in the directory.")
