# This file is to test ideas
import joblib
import streamlit as st
import numpy as np
import pandas as pd

clf = joblib.load('model/model.pkl')

def predict(model, input_arr):
    prediction = model.predict(input_arr)[0]
    return prediction

def run():
    # from PIL import Image
    # image = Image.open('logo.png')
    # image_flower = Image.open('flower.png')

    # st.Image(image, use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
        'How would you like to predict?',
        ('Online', 'Batch')
    )

    st.sidebar.info('This app is created to predict iris specie')
    st.sidebar.success('http://www.github.com/irtizak/iris')

    # st.sidebar.image('flower.png')

    st.title('Iris Specie Prediction App')

    if add_selectbox == 'Online':
        sepal_length = st.number_input('Sepal Length', min_value=4.3, max_value=7.9, value=6.)
        sepal_width = st.number_input('Sepal Width', min_value=2., max_value=4.4, value=3.)
        petal_length = st.number_input('Petal Length', min_value=1., max_value=6.9, value=4.)
        petal_width = st.number_input('Petal Width', min_value=0.1, max_value=2.5, value=2.)

        output=''

        input_arr = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        
        output_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

        if st.button('Predict'):
            output = predict(clf, input_arr)
            output = output_dict[output]
        
        st.success('The specie is {}'.format(output))

    # if add_selection == 'Batch':
    #     file_upload = st.file_uploader('Upload CSV file for predictions', type=['csv'])

    #     if file_upload is not None:
    #         data = pd.read_csv(file_upload)
    #         predictions = predict(model, input_df)
    #         st.write(predictions)

if __name__ == '__main__':
    run()
