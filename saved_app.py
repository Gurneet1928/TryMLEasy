import numpy as np
import pandas as pd
import streamlit as st
# import seaborn as sns
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from streamlit_plotly_events import plotly_events
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA,KernelPCA,FastICA
from sklearn import set_config
import extra_streamlit_components as stx
import io
import mpld3
import streamlit.components.v1 as components
import plotly.graph_objects as go
import tensorflow as tf
import keras

set_config(display="diagram")
st.set_page_config(layout="wide")   # For Wide Web Page
plt.rcParams.update({'font.size': 15, 'font.family': 'serif'})

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
    'Support Vector Classification' : SVR
}


st.markdown(" <h1 style='text-align: center; color: cream;'> TryMLEasy </h1>", unsafe_allow_html=True)  #Main heading
my_bar = st.progress(0, text="Following the Steps>>")   # Progress Bar

# Important variables to be kept safe in session state
# Val -> Value for Step
# got_File -> Stores Dataset
# Model_config -> Stores Model Configuration
# Train_test_data -> Stores training and testing data
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
if "nn_model" not in st.session_state:
    st.session_state.nn_model = None

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        st.write(epoch, "-> ",logs['loss'])


empty_con = st.empty()  # Define an empty container

#----------------------- For dataset Uploading ----------------------------
if st.session_state['val'] == 0:
    with empty_con.container():
        dataset = st.file_uploader("Upload your dataset in a .csv file format.",type=['csv'])   #Upload Files
        button_val = False
        if dataset is None:
            button_val = True
        if st.button(label="Next",disabled=button_val):
            st.session_state['val'] = 1
            st.session_state.got_file = pd.read_csv(dataset)
            st.empty()


#----------------------- For Showing Dataset ----------------------------
if st.session_state['val'] == 1:
    with empty_con.container():
        my_bar.progress(15, text="Analyze the Dataset")
        dataset = st.session_state.got_file
        # ------- To be Modified -------
        #buffer = io.StringIO()
        #dataset.info(buf=buffer)
        #info = buffer.getvalue()
        # -------------------------------

        st.write(dataset)
        if st.button(label="Back",key="analysis_back"):
            st.session_state['val'] = 0 
            st.empty()
        if st.button(label="Next",key="analysis_next"):
            st.session_state['val'] = 2
            st.empty()
# Things to be added in Val = 1
# >> Show If Null/NaN values are present
# >> Show dataset info 
# >> Show options and operations to clean the dataset
            

#----------------------- Feature Selection and Model Details ----------------------------
if st.session_state['val'] == 2:
    with empty_con.container():
        my_bar.progress(65, text="Feature Selection and Model Configuration")

        dataset = st.session_state.got_file
        # ----------------- Options for Users ------------------------
        scaler_step = st.selectbox("Select Preprocessing Step: ",['None','Standard Scaler','MinMax Scaler','Robust Scaler','Normalization'])
        decomp_step = st.selectbox("Select Decomposition Step: ",['None','PCA','Kernal PCA','FastICA'],)
        features = st.multiselect("Select feature values: ",[i for i in dataset.columns])
        target = st.selectbox("Select target column: ",[i for i in dataset.columns][::-1])

        # ------------------ Copy All options in session-state ------------------------
        if target in features:
            st.write("Target exists in features! Try Again")
        elif not features:
            st.write("No Feature Selected!!")
        else:
            st.session_state.model_config['features'] = features
            st.session_state.model_config['target'] = target
            st.session_state.model_config['scaler_step'] = scaler_step
            st.session_state.model_config['decomp_step'] = decomp_step

        # ----------------- Makes sure that target -------------------
        

            # ----------------- Show the Selected value by the user -----------------
        st.write("Selected Features: ",features)
        st.write("Selected Target: ",target)
        st.write("Selected Preprocessing Step: ",scaler_step)
        st.write("Selected Decomposition Step: ",decomp_step)
        button_state = True
            # -----------------------------------------------------------------------

        # A Slider to get the Test Ratio from User
        test_ratio = st.slider("Enter the Test Ratio: ",min_value=1,max_value=50,step=1,value=25)
        
        # Split the dataset
        X_train,X_test,y_train,y_test = train_test_split(dataset[features],dataset[target],test_size=test_ratio/100)

        if st.button(label="Next (Train the Model)",key="train_model") and X_train is not None and target not in features:
            st.session_state['val'] = 3
            #------------------- Copy all the training and testing data into session-state -------------------
            st.session_state.train_test_data['xtrain'] = X_train
            st.session_state.train_test_data['xtest'] = X_test
            st.session_state.train_test_data['ytrain'] = y_train
            st.session_state.train_test_data['ytest'] = y_test
            st.empty()

        if st.button(label="Back",key="model_select_back"):
            st.session_state['val'] = 1
            st.empty()

# Things to be added in Val = 3
# >> More options of models, scaler steps and decomposition steps
# >> Improvement of showing the selected options to users


#----------------------- Option to Select Model Type----------------------------
if st.session_state['val'] == 3:
    with empty_con.container():
        st.write("Select the Type of Model you want to Train")
        my_bar.progress(30, text="Model Selection")
        if st.button(label="Traditional ML Model",key="model_traditional"):
            st.session_state['val'] = 4
            st.empty()
        if st.button(label="Neural Networks",key="model_neurals"):
            st.session_state['val'] = 5
            st.empty()
# Things to be added in Val = 2
# >> Complete Neural Network Part
# >> Change order of code dataset upload -> show dataset -> feature selection -> model selection
            
    
#----------------------- Model Results (For Traditional Models) ----------------------------
if st.session_state.val == 4:
    with empty_con.container():
        my_bar.progress(100, text="Model Results")

        # The below selectbox gets the Model Name from user 
        model_name = st.selectbox("Select The ML Model to Apply: ",['Linear Regression','Logistic Regression','Support Vector Classification'])
        st.session_state.model_config['model_name'] = model_name

        if st.session_state.train_test_data['xtrain'] is not None:
        # ----------------- Make a List to be Used in Pipeline -------
            if st.session_state.model_config['scaler_step'] != 'None':
                ml_pipe.append((st.session_state.model_config['scaler_step'],scale_list[st.session_state.model_config['scaler_step']]()))
            
            if st.session_state.model_config['decomp_step'] != 'None':
                ml_pipe.append((st.session_state.model_config['decomp_step'],decomposition_list[st.session_state.model_config['decomp_step']]()))

            ml_pipe.append((st.session_state.model_config['model_name'],model_list[st.session_state.model_config['model_name']]()))
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

            # ----------------- This section prints the ------------------
            # ----------------- Predicted vs. True Graph -----------------
            max_val = max(max(y_hat),max(ytest))
            min_val = min(min(y_hat),min(ytest))
            fig = go.Figure()
            fig.update_layout(
            autosize=True,
            width=400,
            height=400)
            plt.scatter(y_hat,ytest,c='crimson')
            plt.plot([max_val,min_val],[max_val,min_val],'b-')
            plt.xlabel("Predicted Values")
            plt.ylabel("True Values")
            st.plotly_chart(fig)
            

        if st.button(label="Back",key="feature_back"):
            st.session_state['val'] = 3
            st.empty()

# Things to be added in Val = 4
# >> More metrics to be added
# >> Better graph representations
# >> Interactive Plots
            
if st.session_state.val == 5:
    with empty_con.container():
        my_bar.progress(50, text="Neural Network Configuration")
        
        new_df = pd.DataFrame(columns=['Layer Name','Neurons','Activation Function'])
        cols_config = {
            '_index': st.column_config.TextColumn(default=0),
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

        if st.button(label="Next",key="neural_network_next") :
            st.session_state['val'] = 6
            st.session_state['nn_config'] = new_df
            st.empty()
        if st.button(label="Back",key="neural_network_back"):
            st.session_state['val'] = 3
            st.empty()


if st.session_state['val'] == 6:
    with empty_con.container():
        my_bar.progress(75, text="Neural Network Architecture")

        nn_config = st.session_state['nn_config']
        features = st.session_state.model_config['features']
        target = st.session_state.model_config['target']
        xtrain = st.session_state.train_test_data['xtrain']
        del st.session_state.nn_model

        lss = [(int(nn_config.iloc[i,1]),nn_config.iloc[i,2]) for i in range(len(nn_config))]
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(xtrain.shape[1],)))
        for i in range(len(lss)):
            model.add(tf.keras.layers.Dense(lss[i][0],activation=lss[i][1]))
        model.add(tf.keras.layers.Dense(1))

        model.compile(optimizer='adam', loss='mse')
        #model.build(st.session_state.got_file.shape)
        model.summary(print_fn=lambda x: st.text(x))
        tf.keras.utils.plot_model(model=model, to_file='model_img.jpeg', rankdir='LR',show_shapes=True,expand_nested=True,show_dtype=True,show_layer_names=True,)
        st.image('model_img.jpeg')
        
        if st.button(label="Train",key="nn_train_next"):
            st.session_state['val'] = 7
            model.save('model.keras')
            st.empty()
        
        if st.button(label="Back",key="nn_train_back"):
            st.session_state['val'] = 5
            st.empty()

if st.session_state['val'] == 7:
    with empty_con.container():
        my_bar.progress(100, text="Neural Network Training")
        
        xtrain = st.session_state.train_test_data['xtrain']
        ytrain = st.session_state.train_test_data['ytrain']
        xtest = st.session_state.train_test_data['xtest']
        ytest = st.session_state.train_test_data['ytest']

        model = tf.keras.models.load_model('model.keras')
        history = model.fit(xtrain,ytrain, batch_size=32, epochs=5, callbacks=[CustomCallback()])
        st.write(history.history)

        res = model.evaluate(xtest,ytest,batch_size=32)
        st.write(res)

        loss_res = plt.figure()
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        st.pyplot(loss_res)

        if st.button(label="Back",key="nn_res_back"):
            st.session_state['val'] = 6
            st.empty()
