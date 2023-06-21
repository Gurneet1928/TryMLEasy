import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from streamlit_plotly_events import plotly_events
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA,KernelPCA,FastICA
from sklearn import set_config

set_config(display="diagram")

model_list = {
    'Linear Regression' : LinearRegression,
    'Logistic Regression' : LogisticRegression,
    'Support Vector Classification' : SVC
}

scale_list = {
    'Standard Scaler' : StandardScaler,
    'Normalization' : Normalizer,
    'MinMax Scaler' : MinMaxScaler,
    'Robust Scaler' : RobustScaler
}

decomposition_list = {
    'PCA' : PCA,
    'Kernal PCA' : KernelPCA,
    'FastICA' : FastICA
}

st.markdown(" <h1 style='text-align: center; color: cream;'> TryMLEasy </h1>", unsafe_allow_html=True)

dataset = st.file_uploader("Upload your dataset in a .csv file format.",type=['csv'])
if dataset is not None:
    dataset = pd.read_csv(dataset)
    st.write(dataset.head(5))
    if dataset.isnull().values.any():
        st.write("Null/NaN Values found in dataset")
        st.write("Removing all Null/NaN Values.....")
        dataset = dataset.fillna(dataset.mean)

    features = st.multiselect("Select feature values: ",[i for i in dataset.columns])
    target = st.selectbox("Select target column: ",[i for i in dataset.columns][::-1])
    scaler_step = st.selectbox("Select Preprocessing Step: ",['None','Standard Scaler','MinMax Scaler','Robust Scaler','Normalization'])
    decomp_step = st.selectbox("Select Decomposition Step: ",['None','PCA','Kernal PCA','FastICA'],)
    check_tar_fea = 1
    if target in features:
        st.write("Target exists in features! Try Again")
        check_tar_fea = 0
    if check_tar_fea == 1 and features:
        st.write("Selected Features: ",features)
        st.write("Selected Target: ",target)
        st.write("Selected Preprocessing Step:",scaler_step)
        model_name = st.selectbox("Select The ML Model to Apply: ",['Linear Regression','Logistic Regression','Support Vector Classification'])

        test_ratio = st.slider("Enter the Test Ratio: ",min_value=1,max_value=50,step=1,value=25)
        X_train,X_test,y_train,y_test = train_test_split(dataset[features],dataset[target],test_size=test_ratio/100)
        if X_train is not None:
            model = Pipeline([
                ('scale_type',scale_list[scaler_step]()),
                ('decomp_type',decomposition_list[decomp_step]()),
                ('model_type',model_list[model_name]())],)
            st.write("Pipeline Created: ",model)
            model.fit(X_train,y_train)
            st.write("Model Fitting Successful")
            st.write("Model Score",model.score(X_test,y_test))
            y_hat = model.predict(X_test)
            max_val = max(max(y_hat),max(y_test))
            min_val = min(min(y_hat),min(y_test))
            fig = plt.figure(figsize=(max_val+1,max_val+1))
            plt.scatter(y_hat,y_test)
            plt.plot([max_val,min_val],[max_val,min_val],'b-')
            plt.xlabel("Predicted Values")
            plt.ylabel("True Values")
            st.pyplot(fig)