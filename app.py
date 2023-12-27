import numpy as np
import pandas as pd
import streamlit as st
# import seaborn as sns
from streamlit_plotly_events import plotly_events
from sklearn import set_config
import extra_streamlit_components as stx
import io
import os
import streamlit.components.v1 as components
from streamlit_extras.switch_page_button import switch_page
from st_pages import show_pages,hide_pages,Page

set_config(display="diagram")
st.set_page_config(layout="wide")   # For Wide Web Page

show_pages(
    [
        Page("./app.py","Main Page // Dataset Upload"),
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


# Make a confidential Directory if it does not exist
if not os.path.exists("./confidential"):
    os.mkdir("./confidential")

# -------------------------- Heading -------------------------------
st.markdown(" <h1 style='text-align: center; color: cream;'> TryMLEasy </h1>", unsafe_allow_html=True)  #Main heading

# Important variables to be kept safe in session state
# Val -> Value for Step
# got_File -> Stores Dataset
# Model_config -> Stores Model Configuration
# Train_test_data -> Stores training and testing data
# nn_config -> For Storing Neural Network Configuration in Data Frame
if 'val' not in st.session_state:
    st.session_state.val = 0
if "got_file" not in st.session_state:
    st.session_state.got_file = None
if "model_config" not in st.session_state:
    st.session_state.model_config = {}
if "train_test_data" not in st.session_state:
    st.session_state.train_test_data = {}
if "nn_config" not in st.session_state:
    st.session_state.nn_config = pd.DataFrame(columns=['Layer Name','Neurons','Activation Function'])

#----------------------- For dataset Uploading ----------------------------
dataset = st.file_uploader("Upload your dataset in a .csv file format.",type=['csv'])   #Upload Files
my_bar = st.progress(0, text="Following the Steps>>")   # Progress Bar

# ------------ Button Sate to Ensure File has Been uploaded ----------------
button_val = False
if dataset is None:
    button_val = True

if st.button(label="Next",disabled=button_val):       # Move to Show Dataset
    st.session_state.got_file = pd.read_csv(dataset)
    switch_page("show_dataset")
        