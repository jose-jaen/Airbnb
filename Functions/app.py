import pandas as pd
import xgboost as xgb
import pickle
import streamlit as st
import numpy as np
from math import exp
from scipy import stats

# Read training data
data_link = 'https://raw.githubusercontent.com/jose-jaen/Airbnb/main/Functions/train_data.csv'
data = pd.read_csv(data_link')
data = data.drop('price', axis=1)

# Load AI model
model_name = 'xgb_reg.pkl'
with open(model_name, 'rb') as f:
    xgb_model = pickle.load(f)

# Include Airbnb logo to website and center it
url = 'https://companieslogo.com/img/orig/ABNB_BIG-9ccc2025.png?t=1633511992'
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.image(url)
with col3:
    st.write(' ')

# Set up header and brief description
with st.container():
    st.title('Airbnb Prices Predictor')
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

st.markdown("---")
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

st.markdown('---')

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
    dec, np.mean(data['host_since']), response,
    np.mean(data['host_response_rate']), np.mean(data['host_acceptance_rate']), super_host,
    np.mean(data['host_listings_count']), pic, verified,
    room_type, accommodates, bathrooms, bedrooms, beds,
    min_nights, np.mean(data['maximum_nights']),
    np.mean(data['minimum_nights_avg_ntm']),
    np.mean(data['maximum_nights_avg_ntm']), availability,
    np.mean(data['availability_30']), np.mean(data['availability_90']),
    np.mean(data['availability_365']), np.mean(data['number_of_reviews']),
    np.mean(data['number_of_reviews_ltm']),
    np.mean(data['number_of_reviews_l30d']), np.mean(data['first_review']),
    np.mean(data['last_review']), np.mean(data['review_scores_rating']),
    accuracy, clean, checkin, communication, location, value,
    instant, np.mean(data['calculated_host_listings_count']),
    np.mean(data['calculated_entire']), np.mean(data['calculated_private']),
    np.mean(data['calculated_shared']), stats.mode(data['neighborhood'])[0][0],
    stats.mode(data['neighborhood_group'])[0][0], stats.mode(data['inactive'])[0][0],
    np.mean(data['reviews_month']), stats.mode(data['responds'])[0][0],
    np.mean(data['geo_x']), np.mean(data['geo_y']), np.mean(data['geo_z']),
    stats.mode(data['property'])[0][0], tv, netflix, gym, elevator, fridge,
    heat, hair, air, tub, oven, bbq, cams, workspace, coffee, backyard,
    outdoor, greet, pool, beach, patio, luggage, furniture,
    gender, np.mean(data['sent_median']), np.mean(data['sent_mean']),
    np.mean(data['sent_mode']), no_review)),
    columns=data.columns)

# Center model prediction button
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    run_preds = st.button('Run the model')
    if run_preds:
        st.info(f"Predicted price is ${round(exp(xgb_model.predict(X_test)), 2)}")
with col3:
    st.write(' ')
