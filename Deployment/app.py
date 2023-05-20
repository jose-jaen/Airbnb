import pandas as pd
import xgboost as xgb
import pickle
import streamlit as st
import numpy as np
from math import exp
from scipy import stats
from io import BytesIO
import requests
from sentiment import *

# Set up header and brief description
with st.container():
    st.title('Airbnb Price Predictor')
    st.markdown('Simulate real estate market asset valuations with AI!')
    st.markdown('Provide data about your Airbnb listing and get predictions!')

# Begin new section for listings features
st.markdown('---')
st.subheader('Listing characteristics')
col1, col2 = st.columns(2)
with col1:
    accommodates = st.slider('Maximum Capacity', 1, 16, 4)
    bathrooms = st.slider('Number of bathrooms', 1, 9, 2)
    room_type = st.selectbox('Room Type',
                             ('Private room', 'Entire apartment', 'Shared room', 'Hotel room'))
    instant = st.selectbox('Can the listing be instantly booked?',
                           ('No', 'Yes'))
with col2:
    beds = st.slider('Number of beds', 1, 32, 2)
    bedrooms = st.slider('Number of bedrooms', 1, 24, 2)
    min_nights = st.slider('Minimum number of nights', 1, 20, 3)
    amenities = st.multiselect(
        'Select available amenities',
        ['TV', 'Wifi', 'Netflix', 'Swimming pool', 'Hot tub', 'Gym', 'Elevator',
         'Fridge', 'Heating', 'Air Conditioning', 'Hair dryer', 'BBQ', 'Oven',
         'Security cameras', 'Workspace', 'Coffee maker', 'Backyard',
         'Outdoor dining', 'Host greeting', 'Beachfront', 'Patio',
         'Luggage dropoff', 'Furniture'],
        ['TV', 'Wifi'])

# Section for host info
st.markdown('---')
st.subheader('Host Information')
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox('Host gender', ('Female', 'Male', 'Other/Corporation'))
    pic = st.selectbox('Does your host have a profile picture?', ('Yes', 'No'))
    dec = st.selectbox('Did your host write a description about the listing?', ('Yes', 'No'))
    super_host = st.selectbox('Is your host a superhost?', ('No', 'Yes'))
with col2:
    verified = st.selectbox('Is your host verified?', ('Yes', 'No'))
    availability = st.selectbox('Is the listing available?', ('Yes', 'No'))
    response = st.selectbox('Response rate', (
        'Within an hour', 'Within a few hours', 'Within a day', 'Within a few days'))
    no_review = st.selectbox('Did your host get any review?', ('Yes', 'No'))

host_since = st.slider(
    'Number of days your host has been using Airbnb',
    1, 5000, 2000)

st.markdown('---')
st.subheader("Guests' feedback")
col1, col2, col3 = st.columns(3)
with col1:
    location = st.slider('Location rating', 1.0, 5.0, 4.0, step=0.5)
    checkin = st.slider('Checkin rating', 1.0, 5.0, 3.0, step=0.5)
with col2:
    clean = st.slider('Cleanliness rating', 1.0, 5.0, 3.0, step=0.5)
    communication = st.slider('Communication rating', 1.0, 5.0, 4.0, step=0.5)
with col3:
    value = st.slider('Value rating', 1.0, 5.0, 3.5, step=0.5)
    accuracy = st.slider('Accuracy rating', 1.0, 5.0, 4.2, step=0.5)

# Center model prediction button
_, col2, _ = st.columns(3)
with col2:
    run_preds = st.button('Run the model')
    if run_preds:    
        # Load AI model
        with open('Deployment/xgb_reg.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
            
        # One-hot encoding amenities
        options = ['TV', 'Wifi', 'Netflix', 'Swimming pool', 'Hot tub', 'Gym', 'Elevator',
                   'Fridge', 'Heating', 'Air Conditioning', 'Hair dryer', 'BBQ', 'Oven',
                   'Security cameras', 'Workspace', 'Coffee maker', 'Backyard',
                   'Outdoor dining', 'Host greeting', 'Beachfront', 'Patio',
                   'Luggage dropoff', 'Furniture']

        amens = [1 if i in amenities else 0 for i in options]
        tv, wifi, netflix, pool = amens[0], amens[1], amens[2], amens[3]
        tub, gym, elevator, fridge = amens[4], amens[5], amens[6], amens[7]
        heat, air, hair, bbq = amens[8], amens[9], amens[10], amens[11]
        oven, cams, workspace, coffee = amens[12], amens[13], amens[14], amens[15]
        backyard, outdoor, greet, beach = amens[16], amens[17], amens[18], amens[19]
        patio, luggage, furniture = amens[20], amens[21], amens[22]

        # One-hot encoding binary features
        dec = 1 if dec == 'Yes' else 0
        super_host = 1 if super_host == 'Yes' else 0
        pic = 1 if pic == 'Yes' else 0
        verified = 1 if verified == 'Yes' else 0
        availability = 1 if availability == 'Yes' else 0
        instant = 1 if instant == 'Yes' else 0
        gender = 1 if gender == 'Yes' else 0
        no_review = 0 if no_review == 'Yes' else 1

        # Encode room_type feature
        rooms = {
            'Private room': 1,
            'Entire home/apt': 2,
            'Shared room': 3,
            'Hotel room': 4
        }
        room_type = rooms.get(room_type)

        # Encode response_time feature
        responses = {
            'Within an hour': 1,
            'Within a few hours': 2,
            'Within a day': 3,
            'Within a few days': 4
        }
        response = responses.get(response)

        # Set up feature matrix for predictions
        X_test = pd.DataFrame(data=np.column_stack((
            dec, 2049.8854, response, 76.6324, 
            71.2640, super_host, 69.1362, pic, verified,
            room_type, accommodates, bathrooms, 
            bedrooms, beds, min_nights, 601.9025,
            27.1608, 332469.8321, availability,
            9.7215, 40.2549, 179.5286, 36.1675,
            9.8272, 0.9183, 827.2542, 238.0988, 
            4.6789, accuracy, clean, checkin, 
            communication, location, value,
            instant, 18.3119, 14.5421, 3.3295,
            0.3858, 0, 114, 0, 1.1963, 1,
            -0.3559, -0.7283, 0.5242, 17, 
            tv, netflix, gym, elevator, fridge,
            heat, hair, air, tub, oven, bbq, cams, 
            workspace, coffee, backyard, outdoor, 
            greet, pool, beach, patio, luggage, furniture,
            gender, 0.9643, 0.9029, 0.9650, no_review)))
        
        st.info(f"Predicted price is ${round(exp(xgb_model.predict(X_test)), 2)}")
 
st.markdown('---')
st.subheader('Sentiment Analysis')
st.markdown('Write a review and get the predicted sentiment!')

if 'disabled' not in st.session_state:
    st.session_state['disabled'] = False

    
def disable():
    st.session_state['disabled'] = True

    
user_input = st.text_input(
    'Introduce your own review!', 
    disabled=st.session_state.disabled, 
    on_change=disable
)

run_sent = st.button('Estimate sentiment')

if run_sent:
    # Load transformer for sentiment analysis
    model, tokenizer, config = load_model()
    text = preprocess(user_input)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Print labels and scores
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    sentiment = config.id2label[ranking[0]]
    st.info(f'Predicted sentiment is {sentiment}')

st.markdown('---')
st.subheader('About')
st.markdown('This a Data Science project unaffiliated with Airbnb')
st.markdown('Note that the predicted price is the amount hosts charge **per night**!')
st.markdown('Prediction accuracy is limited to listings in **Los Angeles** from **summer 2022**')
st.markdown('Sentiment Analysis prediction is restricted to one request due to limited compute resources')
transformer = 'https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest'
st.markdown('The deployed NLP model is the transformer [RoBERTa](%s)' % transformer)
thesis = 'https://github.com/jose-jaen/Airbnb'
st.markdown('Feel free to check the entirety of my Bachelor Thesis [here](%s)' % thesis)
linkedin = 'https://www.linkedin.com/in/jose-jaen/'
st.markdown('Reach out to [José Jaén Delgado](%s) for any questions' % linkedin)
