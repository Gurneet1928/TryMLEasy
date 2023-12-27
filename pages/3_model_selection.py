import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from st_pages import show_pages,hide_pages,Page

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
st.markdown(" <h2 style='text-align: center; color: cream;'> Select the Type of Model you want to Train </h2>", unsafe_allow_html=True) 

#----------------------- Option to Select Model Type----------------------------
if st.button(label="Traditional ML Model",key="model_traditional"):
    switch_page("traditional model")

if st.button(label="Neural Networks",key="model_neurals"):
    switch_page('neural network configuration')

if st.button(label="Back (Feature Selection)",key="feature_selection"):
    switch_page("feature selection")