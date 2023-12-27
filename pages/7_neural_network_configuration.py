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
])

if 'keras_model' not in st.session_state:
    st.session_state.keras_model = None
if 'model_archi' not in st.session_state:
    st.session_state.model_archi = None
# -------------------------------------------------------------------

# -------------------------- Heading -------------------------------
st.markdown(" <h2 style='text-align: center; color: cream;'> Neural Network Setup </h2>", unsafe_allow_html=True) 

# # -------------------- Small Disclaimer for User --------------------
st.write("Please enter the index value as you add any layer. Otherwise, non-indexed layers will not be accepted!!")
st.write("We are working on this. Please support !!")

# Show Dynamic Dataframe for User Interface
new_df = pd.DataFrame(columns=['Layer Name','Neurons','Activation Function'])
cols_config = {
    '_index': st.column_config.NumberColumn(required=True),
    'Layer Name': st.column_config.TextColumn(default='A Happy Layer',required=True,),
    'Neurons': st.column_config.NumberColumn(
        'neurons',
        help='Select the number of neurons',
        width="medium",
        default=10,
        min_value=2,
        max_value=100,
        step=1,
        required=True,
        ),
    "Activation Function": st.column_config.SelectboxColumn(
        "activation_function",
        width="medium",
        options=[
            "relu",
            "sigmoid",
            "tanh",
        ],
        required=True,
        default='relu',
        )
    }

new_df = st.data_editor(new_df,
                        column_config=cols_config, 
                        num_rows="dynamic",
                        use_container_width=True,
                        )

if st.button(label="Verify Architecture",key="verify_architecture"):        # Verify Architecture Button
    st.session_state.nn_config = new_df
    switch_page("neural_network_architecture")

if st.button(label="Back (Model Selection)",key="back_model_selection"):    # Back Button
    switch_page('model_selection')