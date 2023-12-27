import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from st_pages import show_pages,hide_pages,Page

# -------------------- Show Pages and Hide Pages --------------------
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

# -------------------------- Heading -------------------------------
st.markdown(" <h2 style='text-align: center; color: cream;'> Is this how Your dataset Looks ? </h2>", unsafe_allow_html=True) 

#----------------------- For Showing Dataset ---------------------------

dataset = st.session_state.got_file

# ------- To be Modified -------
#buffer = io.StringIO()
#dataset.info(buf=buffer)
#info = buffer.getvalue()
# -------------------------------

st.dataframe(dataset, use_container_width=True)

if st.button(label="Back (Dataset Upload)",key="analysis_back"):
    switch_page("app") 
    
if st.button(label="Next (Feature Selection)",key="analysis_next"):
    switch_page("feature_selection")