import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Page config
st.set_page_config(page_title="Loan Default Prediction Demo", layout="wide")

# Title
st.title("Loan Default Prediction Analysis")
st.write("This is a demo application showing feature importance in loan default prediction")

def generate_dummy_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate features
    data = {
        'income': np.random.normal(50000, 20000, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'loan_amount': np.random.normal(200000, 100000, n_samples),
    }
    
    # Generate categorical features
    locations = ['Urban', 'Suburban', 'Rural']
    education = ['High School', 'Bachelor', 'Master', 'PhD']
    
    data['location'] = np.random.choice(locations, n_samples)
    data['education'] = np.random.choice(education, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable (default_rate)
    prob_default = 1 / (1 + np.exp(-(
        -5 
        - df['income']/50000 
        + df['loan_amount']/100000 
        - df['credit_score']/200
        + (df['age'] < 25).astype(int)
    )))
    
    df['default'] = np.random.binomial(1, prob_default)
    
    return df

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = LabelEncoder()

# Generate Data button
if st.button("Generate New Data"):
    st.session_state.df = generate_dummy_data(1000)
    
    # Prepare data for modeling
    df_model = st.session_state.df.copy()
    df_model['location'] = st.session_state.label_encoder.fit_transform(df_model['location'])
    df_model['education'] = st.session_state.label_encoder.fit_transform(df_model['education'])
    
    X = df_model.drop('default', axis=1)
    y = df_model['default']
    
    # Train Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.session_state.model = RandomForestClassifier(n_estimators=100, random_state=42)
    st.session_state.model.fit(X_train, y_train)

# Only show the rest of the app if data has been generated
if st.session_state.df is not None:
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(st.session_state.df.head())

    # Basic statistics
    st.subheader("Basic Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Default Rate Distribution")
        default_dist = st.session_state.df['default'].value_counts(normalize=True)
        fig = px.pie(values=default_dist.values, names=default_dist.index, 
                     title="Default Rate Distribution")
        st.plotly_chart(fig)

    with col2:
        st.write("Average Income by Default Status")
        avg_income = st.session_state.df.groupby('default')['income'].mean()
        fig = px.bar(x=avg_income.index, y=avg_income.values, 
                     title="Average Income by Default Status")
        st.plotly_chart(fig)

    # Feature Importance Analysis
    st.subheader("Feature Importance Analysis")
    
    # Feature importance plot
    importance_df = pd.DataFrame({
        'feature': st.session_state.df.drop('default', axis=1).columns,
        'importance': st.session_state.model.feature_importances_
    }).sort_values('importance', ascending=True)

    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                 title="Feature Importance in Predicting Loan Default")
    st.plotly_chart(fig)

    # Interactive prediction
    st.subheader("Try Predicting Default Risk")
    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input("Income", min_value=0, value=50000)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        employment_years = st.number_input("Years of Employment", min_value=0, value=5)

    with col2:
        loan_amount = st.number_input("Loan Amount", min_value=0, value=200000)
        location = st.selectbox("Location", ['Urban', 'Suburban', 'Rural'])
        education = st.selectbox("Education", ['High School', 'Bachelor', 'Master', 'PhD'])

    if st.button("Predict Default Risk"):
        # Prepare input data
        input_data = pd.DataFrame({
            'income': [income],
            'age': [age],
            'credit_score': [credit_score],
            'employment_years': [employment_years],
            'loan_amount': [loan_amount],
            'location': [st.session_state.label_encoder.transform([location])[0]],
            'education': [st.session_state.label_encoder.transform([education])[0]]
        })
        
        # Make prediction
        prediction = st.session_state.model.predict_proba(input_data)[0][1]
        
        st.write(f"Predicted Default Risk: {prediction:.2%}")
else:
    st.info("Please click 'Generate New Data' to start the analysis") 