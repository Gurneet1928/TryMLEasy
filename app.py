import numpy as np
import pandas as pd
import streamlit as st
# import seaborn as sns
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

ml_pipe = []    # Empty list to store the ML Techniques and Models. To be used inside the pipeline

# List of Supported Scaling techniques
scale_list = {
    'Standard Scaler' : StandardScaler,
    'Normalization' : Normalizer,
    'MinMax Scaler' : MinMaxScaler,
    'Robust Scaler' : RobustScaler
}

# List of Supported Decomposition Techniques
decomposition_list = {
    'PCA' : PCA,
    'Kernal PCA' : KernelPCA,
    'FastICA' : FastICA
}

# List of Models to call 
model_list = {
    'Linear Regression' : LinearRegression,
    'Logistic Regression' : LogisticRegression,
    'Support Vector Classification' : SVC
}


st.markdown(" <h1 style='text-align: center; color: cream;'> TryMLEasy </h1>", unsafe_allow_html=True)  #Main heading

dataset = st.file_uploader("Upload your dataset in a .csv file format.",type=['csv'])   #Upload Files
if dataset is not None:
    dataset = pd.read_csv(dataset)
    st.write(dataset.head(5))

    # ----------------- Working on this Section -----------------
    # ----------------- What does this Section Do ? -------------
    # Check for Null/NaN values in dataset, if present, prompt user
    # to take action such as remove them or fill them
    if dataset.isnull().values.any():
        st.write("Null/NaN Values found in dataset")
        st.write("Removing all Null/NaN Values.....")
        dataset = dataset.fillna(dataset.mean)


    # ------------------------------------------------------------

    # ----------------- Options for Users ------------------------
    features = st.multiselect("Select feature values: ",[i for i in dataset.columns])
    target = st.selectbox("Select target column: ",[i for i in dataset.columns][::-1])
    scaler_step = st.selectbox("Select Preprocessing Step: ",['None','Standard Scaler','MinMax Scaler','Robust Scaler','Normalization'])
    decomp_step = st.selectbox("Select Decomposition Step: ",['None','PCA','Kernal PCA','FastICA'],)
    # ------------------------------------------------------------

    check_tar_fea = 1   # A variable with default value 1. In case the target exists in feature list, prompt user and value becomes 1.

    # ----------------- Makes sure that target -------------------
    # ----------------- does not exits in feature list -----------
    if target in features:
        st.write("Target exists in features! Try Again")
        check_tar_fea = 0
    if check_tar_fea == 1 and features:

        # ----------------- Show the Selected value by the user -----------------
        st.write("Selected Features: ",features)
        st.write("Selected Target: ",target)
        st.write("Selected Preprocessing Step:",scaler_step)
        # -----------------------------------------------------------------------

        # The below selectbox gets the Model Name from user 
        model_name = st.selectbox("Select The ML Model to Apply: ",['Linear Regression','Logistic Regression','Support Vector Classification'])

        # A Slider to get the Test Ratio from User
        test_ratio = st.slider("Enter the Test Ratio: ",min_value=1,max_value=50,step=1,value=25)

        # Split the dataset
        X_train,X_test,y_train,y_test = train_test_split(dataset[features],dataset[target],test_size=test_ratio/100)

        # This check makes sure that in case target/features is None, the program does not proceed forward and raise error.
        if X_train is not None:

            # ----------------- Make a List to be Used in Pipeline -------
            if scaler_step is not 'None':
                ml_pipe.append((scaler_step,scale_list[scaler_step]()))
            
            if decomp_step is not 'None':
                ml_pipe.append((decomp_step,decomposition_list[decomp_step]()))

            ml_pipe.append((model_name,model_list[model_name]()))
            #-------------------------------------------------------------

            # Create a Pipeline with the required specifications
            model = Pipeline(ml_pipe)
            st.write("Pipeline Created: ",model)

            # Fit the Model using X_train and Y_Train
            model.fit(X_train,y_train)
            st.write("Model Fitting Successful")

            # Print the model score from X_test and y_test
            st.write("Model Score",model.score(X_test,y_test))

            # Create a list of predicted values of X_test
            y_hat = model.predict(X_test)

            # ----------------- This section prints the ------------------
            # ----------------- Predicted vs. True Graph -----------------
            max_val = max(max(y_hat),max(y_test))
            min_val = min(min(y_hat),min(y_test))
            fig = plt.figure(figsize=(max_val+1,max_val+1))
            plt.scatter(y_hat,y_test)
            plt.plot([max_val,min_val],[max_val,min_val],'b-')
            plt.xlabel("Predicted Values")
            plt.ylabel("True Values")
            st.pyplot(fig)


