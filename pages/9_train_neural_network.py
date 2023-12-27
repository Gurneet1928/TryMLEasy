import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
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

# -------------------------------------------------------------------


# -------------------------- Heading -------------------------------
st.markdown(" <h2 style='text-align: center; color: cream;'> Metrics and Graphs </h2>", unsafe_allow_html=True) 

# Custom Callback for showing Loss and Val_Loss at each epoch
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        st.write("{epc} ----- {loss:.2f} ----- {val:.2f}".format(epc=epoch,loss=logs['loss'],val=logs['val_loss']))

# Get all Training and Testing Data
xtrain = st.session_state.train_test_data['xtrain']
ytrain = st.session_state.train_test_data['ytrain']
xtest = st.session_state.train_test_data['xtest']
ytest = st.session_state.train_test_data['ytest']

# Get Epochs and Batch Size from User
epc = st.number_input("Enter the epochs you want to train for : ", step = 1)
btc = st.number_input("Enter the Batch Size : ", step = 1)

# Load the Model
#model = tf.keras.models.load_model('./confidential/model.keras',compile=True)
model = st.session_state.keras_model
# Button to check that epochs and batch sizez are not 0
button_state = False
if btc == 0 or epc == 0:
    button_state = True
else:
    button_state = False

if st.button(label="Train Neural Network",disabled=button_state):
    st.write("Epoch ---- Loss ---- Val_Loss")

    # Fit the Model
    history = model.fit(xtrain,ytrain,validation_data=(xtest,ytest), batch_size=btc, epochs=epc, callbacks=[CustomCallback()])

    # Evaluate the Model on the test data one final time
    res = model.evaluate(xtest,ytest,batch_size=32)
    st.write("Final Loss on Test Data : {}".format(res[0]))

    # Plot the Training v/s Validation Loss Graph
    loss_res = plt.figure(figsize=(13,5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train_loss','val_loss'], loc='upper right')
    st.pyplot(loss_res)

if st.button(label="Back (Architecture View)",key="nn_res_back"):       # Back Button 
    switch_page("neural network architecture")
    
