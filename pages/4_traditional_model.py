# -------------------- Few Random Libraries -------------------
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from st_pages import show_pages,hide_pages,Page
# ------------------------ Libraries for ML ----------------------
from sklearn.linear_model import LinearRegression,LogisticRegression,SGDClassifier, Ridge, LassoLars
from sklearn.naive_bayes import  GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler,RobustScaler
from sklearn.decomposition import PCA,KernelPCA,FastICA
# ------------------------- Metrics List ----------------------------
from sklearn.metrics import mean_squared_error, explained_variance_score, max_error, accuracy_score, f1_score, precision_score, recall_score


# -------------------- Show Pages and Hide Pages --------------------
show_pages(
    [
        Page("./home_page.py","Main Page // Dataset Upload"),
        Page("./pages/1_show_dataset.py","Show Dataset"),
        Page("./pages/2_feature_selection.py","Feature Selection"),
        Page("./pages/3_model_selection.py","Model Selection"),
        Page("./pages/4_traditional_model.py","Traditional ML Model"),
        Page("./pages/7_neural_network_configuration.py","Neural Network Configurations"),
        Page("./pages/8_neural_network_architecture.py","Neural Network Architecture"),
        Page("./pages/9_train_neural_network.py","Training Neural Networks"),
    ]
)
hide_pages([
    'Neural Network Configurations',
    'Neural Network Architecture',
    'Training Neural Networks',
])
# ---------------------------------------------------------------

plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})          # Matplotlib Figures

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

# List of Models to call for each Classification and Regression
model_list = { 
    'Classification' :{ 'Logistic Regression' : LogisticRegression,
                        'Decision Tree Classifier' : DecisionTreeClassifier,
                        'Gaussian Naive Bayes' : GaussianNB,
                        'Support Vector Classification' : SVC,
                        'K-Nearest Neighbors': KNeighborsClassifier,
                        'Stochastic Gradient Descent Classifier' : SGDClassifier },
    'Regression' : {'Linear Regression' : LinearRegression,
                    'Support Vector Regressor' : SVR,
                    'Rigde Regression' : Ridge,
                    'Least Angle Regression': LassoLars,}
}
# ---------------------------------------------------------------

# Metrics List for Classification and Regression 
metrics = {
    'Regression' : {'Mean Squared Error' : mean_squared_error,
                    'Explained Variance' : explained_variance_score,
                    'Max Error' : max_error},
    'Classification' : {'Accuracy' : accuracy_score,
                        'F1 Score' : f1_score,
                        'Precision' : precision_score,
                        'Recall' : recall_score,}
}

# ---------------------------------------------------------------

ml_pipe = []            # Empty List for ML Pipe


# -----------------------------------------------------------------

# -------------------------- Heading -------------------------------
st.markdown(" <h2 style='text-align: center; color: cream;'> Traditional Model Setup </h2>", unsafe_allow_html=True) 


#----------------------- (For Traditional Models) ----------------------------

# The below selectbox gets the Model Type and Name from user 
model_type = st.selectbox("Select the Supervised Model Type: ",['Classification','Regression'])
if model_type == 'Classification':
    model_name = st.selectbox("Select The ML Model to Apply: ",model_list['Classification'].keys())
else:
    model_name = st.selectbox("Select The ML Model to Apply: ",model_list['Regression'].keys())

if st.button("Train Model on {}".format(model_name)):           # Button to Train the Model on Click
    st.session_state.model_config['model_name'] = model_name

    # ----------------- Make a List to be Used in Pipeline -------

    # For Scaling Technique >
    if st.session_state.model_config['scaler_step'] != 'None':
        ml_pipe.append((st.session_state.model_config['scaler_step'],scale_list[st.session_state.model_config['scaler_step']]()))
    
    # For Decomposition Technique >
    if st.session_state.model_config['decomp_step'] != 'None':
        ml_pipe.append((st.session_state.model_config['decomp_step'],decomposition_list[st.session_state.model_config['decomp_step']]()))

    # Append Model Name into pipeline >
    ml_pipe.append((st.session_state.model_config['model_name'],model_list[model_type][st.session_state.model_config['model_name']]()))
    #-------------------------------------------------------------

    # Create a Pipeline with the required specifications
    model = Pipeline(ml_pipe)
    st.write("Pipeline Created: ",model)

    # Copy all values of train and test data from session-state to variables
    xtrain = st.session_state.train_test_data['xtrain']
    ytrain = st.session_state.train_test_data['ytrain']
    xtest = st.session_state.train_test_data['xtest']
    ytest = st.session_state.train_test_data['ytest']

    # Fit the Model using X_train and Y_Train
    model.fit(xtrain,ytrain)
    st.write("Model Fitting Successful")

    # Print the model score from X_test and y_test
    st.write("Model Score",model.score(xtest,ytest))

    # Create a list of predicted values of X_test
    y_hat = model.predict(xtest)

    # ----------------- This section prints the Results------------------
    if model_type == 'Classification':
        try:
            for i in metrics['Classification']:
                st.write("{}: {}".format(i,metrics['Classification'][i](y_hat,ytest)))
        except:
            st.warning("Metric Calculation Error !! Probably Because Classification used on Discrete Data !!")
            st.error("Regression Models Recommended")
    else:
        for i in metrics['Regression']:
            st.write("{}: {}".format(i,metrics['Regression'][i](y_hat,ytest)))
    

    # ----------------- Predicted vs. True Graph -----------------
    max_val = max(max(y_hat),max(ytest))
    min_val = min(min(y_hat),min(ytest))
    fig = plt.figure(figsize=(12,12))
    plt.scatter(y_hat,ytest,c='crimson')
    plt.plot([max_val,min_val],[max_val,min_val],'b-')
    plt.xlabel("Predicted Values")
    plt.ylabel("True Values")
    st.pyplot(fig)


if st.button(label="Back (Model Selection)",key="model_neurals"):
    switch_page('model_selection')
