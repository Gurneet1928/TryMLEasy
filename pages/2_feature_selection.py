import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from sklearn.model_selection import train_test_split
from st_pages import show_pages,hide_pages,Page
import seaborn as sns
import matplotlib.pyplot as plt
import PIL

plt.rcParams['figure.figsize'] = 2,2            # Figure Size for Matplotlib

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
    'Traditional ML Model',
    'Neural Network Configurations',
    'Neural Network Architecture',
    'Training Neural Networks',
])


# -------------------------------------------------------------------

# -------------------------- Heading -------------------------------
st.markdown(" <h2 style='text-align: center; color: cream;'> Please Select Feature, Target and Preprocessing Steps </h2>", unsafe_allow_html=True) 

#----------------------- Feature Selection and Model Details ----------------------------

dataset = st.session_state.got_file
fig = plt.figure(figsize=(10, 10))
plt.title("Correlation Matrix For Feature Analysis")
ax = sns.heatmap(dataset.corr())
st.write(fig)

# ----------------- Options for Users ------------------------
st.write("/\>> Preprocessing Step and Decomposition Step are currently only supported for use by Traditional ML Models <<")
scaler_step = st.selectbox("Select Preprocessing Step: ",['None','Standard Scaler','MinMax Scaler','Robust Scaler','Normalization'])
decomp_step = st.selectbox("Select Decomposition Step: ",['None','PCA','Kernal PCA','FastICA'],)
features = st.multiselect("Select feature values*: ",[i for i in dataset.columns])
target = st.selectbox("Select target column*: ",[i for i in dataset.columns][::-1])

button_state = False
# ------------------ Copy All options in session-state ------------------------
if target in features:
    st.error("Target exists in features! Try Again")
    button_state = True
elif not features or None in features:
    st.error("No Feature Selected!!")
    button_state = True
else:
    st.session_state.model_config['features'] = features
    st.session_state.model_config['target'] = target
    st.session_state.model_config['scaler_step'] = scaler_step
    st.session_state.model_config['decomp_step'] = decomp_step
    button_state = False


# ----------------- Show the Selected value by the user -----------------
st.write("Selected Features: ",features)
st.write("Selected Target: ",target)
st.write("Selected Preprocessing Step: ",scaler_step)
st.write("Selected Decomposition Step: ",decomp_step)
# -----------------------------------------------------------------------

# A Slider to get the Test Ratio from User
test_ratio = st.slider("Enter the Test Ratio: ",min_value=1,max_value=50,step=1,value=25)


# Split the dataset
X_train,X_test,y_train,y_test = train_test_split(dataset[features],dataset[target],test_size=test_ratio/100)


if st.button(label="Next (Model Type Selection)",key="train_model",disabled=button_state) :
    #------------------- Copy all the training and testing data into session-state -------------------
    st.session_state.train_test_data['xtrain'] = X_train
    st.session_state.train_test_data['xtest'] = X_test
    st.session_state.train_test_data['ytrain'] = y_train
    st.session_state.train_test_data['ytest'] = y_test
    switch_page("model_selection")    


if st.button(label="Back (Dataset View)",key="model_select_back"):
    switch_page("show_dataset")

