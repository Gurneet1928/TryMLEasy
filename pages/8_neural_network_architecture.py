import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from st_pages import show_pages,hide_pages,Page
import os

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
])

# -------------------------------------------------------------------

# -------------------------- Heading -------------------------------
st.markdown(" <h2 style='text-align: center; color: cream;'> Neural Network Configuration Verification </h2>", unsafe_allow_html=True) 


nn_config = st.session_state['nn_config']
features = st.session_state.model_config['features']
target = st.session_state.model_config['target']
xtrain = st.session_state.train_test_data['xtrain']

lss = [(int(nn_config.iloc[i,1]),nn_config.iloc[i,2]) for i in range(len(nn_config))]
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(xtrain.shape[1],)))
for i in range(len(lss)):
    model.add(tf.keras.layers.Dense(lss[i][0],activation=lss[i][1]))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.summary(print_fn=lambda x: st.text(x))

# -------- Needs to be worked upon -------------
#st.session_state.model_archi = tf.keras.utils.plot_model(model=model, to_file='./confidential/model_img.jpeg', rankdir='LR',show_shapes=True,expand_nested=True,show_dtype=True,show_layer_names=True,)
#st.write(st.session_state.model_archi)

if st.button(label="Train Model",key="nn_train_next"):
    st.session_state.keras_model = model
    switch_page("training neural networks")

if st.button(label="Back (Neural Network Configuration)",key="nn_train_back"):
    switch_page("neural network configuration")
