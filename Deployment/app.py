import pandas as pd
import xgboost as xgb
import pickle
import streamlit as st
import numpy as np
from math import exp
from scipy import stats
from io import BytesIO
import requests

# Load AI model
with st.spinner('Downloading AI model, please wait'):
    model_link = 'https://github.com/jose-jaen/Airbnb/blob/main/Deployment/xgb_reg.pkl?raw=true'
    mfile = BytesIO(requests.get(model_link).content)
    xgb_model = pickle.load(mfile)

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

# Center model prediction button
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    run_preds = st.button('Run the model')
    if run_preds:    
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
            dec, 2049.8854292157043, response,
            76.63240972415284, 71.26407808976616, super_host,
            69.13628641151577, pic, verified,
            room_type, accommodates, bathrooms, bedrooms, beds,
            min_nights, 601.9025422419918,
            27.160806845210516,
            332469.8321718717, availability,
            9.72158897846971, 40.25490377783956,
            179.52868130849782, 36.1675779198715,
            9.827263460290983,
            0.9183887807741019, 827.2542241991783,
            238.09884780526983, 4.678916998733513,
            accuracy, clean, checkin, communication, location, value,
            instant, 18.311988385382882,
            14.542180211904983, 3.329502980879128,
            0.3858462298829271, 0,
            114, 0,
            1.1963299663299665, 1,
            -0.35596108151187583, -0.7283118417599816, 0.5242905694820023,
            17, tv, netflix, gym, elevator, fridge,
            heat, hair, air, tub, oven, bbq, cams, workspace, coffee, backyard,
            outdoor, greet, pool, beach, patio, luggage, furniture,
            gender, 0.9643530102245699, 0.9029613928046442,
            0.9650943687640935, no_review)),
            columns=['description', 'host_since', 'host_response_time', 'host_response_rate',
               'host_acceptance_rate', 'host_is_superhost', 'host_listings_count',
               'host_has_profile_pic', 'host_identity_verified', 'room_type',
               'accommodates', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights',
               'maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm',
               'has_availability', 'availability_30', 'availability_90',
               'availability_365', 'number_of_reviews', 'number_of_reviews_ltm',
               'number_of_reviews_l30d', 'first_review', 'last_review',
               'review_scores_rating', 'review_scores_accuracy',
               'review_scores_cleanliness', 'review_scores_checkin',
               'review_scores_communication', 'review_scores_location',
               'review_scores_value', 'instant_bookable',
               'calculated_host_listings_count', 'calculated_entire',
               'calculated_private', 'calculated_shared', 'neighborhood',
               'neighborhood_group', 'inactive', 'reviews_month', 'responds', 'geo_x',
               'geo_y', 'geo_z', 'property', 'tv', 'netflix', 'gym', 'elevator',
               'fridge', 'heating', 'hair_dryer', 'air_conditioning', 'hot_tub',
               'oven', 'bbq', 'security cameras', 'workspace', 'coffee', 'backyard',
               'outdoor_dining', 'greets', 'pool', 'beachfront', 'patio', 'luggage',
               'furniture', 'nlp_gender', 'sent_median', 'sent_mean', 'sent_mode',
               'no_review'])
        st.info(f"Predicted price is ${round(exp(xgb_model.predict(X_test)), 2)}")
with col3:
    st.write(' ')
