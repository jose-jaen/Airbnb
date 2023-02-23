# Airbnb Price Prediction :house:

Looking for a fair-priced Airbnb listing to book? 

Are you a real estate firm or Airbnb host seeking to set competitive prices?

Artificial Intelligence is surely the way to go! :robot:

In this Quantitative Economics Bachelor Thesis, Machine Learning and Deep Learning algorithms are used to accurately predict Airbnb rental prices.

Two datasets for Los Angeles city are retrieved from [Inside Airbnb](http://insideairbnb.com/): listings data and the reviews dataset.

Combining custom functions, transfer learning and Open Source AI frameworks such as scikit-learn, TensorFlow and PyTorch, several AI models are built to help economic agents make informed decisions.

A Bayesian perspective is taken to perform statistical inference, hyperparameter tuning and data modeling.

Additionally, XAI methods are used to overcome the black box problem of AI algorithms.

# Price Prediction App
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airbnb-prices.streamlit.app/)

Click on the icon above to acces the Airbnb Price Prediction Web App!

# Bachelor Thesis Document

The link below gives access to the thesis explaining all Machine Learning, XAI and Statistical Inference techniques. 

- [Thesis Document](https://github.com/jose-jaen/Airbnb/blob/main/Project/Thesis.pdf)

# General Code

Since Python code has been divided into multiples files, the one below combined all the programmed functions for retrieving relevant results.

- [Project Code](https://github.com/jose-jaen/Airbnb/blob/main/Project/airbnb_project.py)

# Preprocessing and Responsible AI

From creating and modifying features to opening the black box problem of AI, all Data Mining algorithms can be found in the following link.

- [Data Cleaning, Feature Engineering & XAI](https://github.com/jose-jaen/Airbnb/blob/main/Functions/general_functions.py)

# NLP and CV

NLP and Computer Vision algorithms used on the reviews and listings dataset, respectively. 

VADER, a sentiment analysis tool was tweaked so as to adapt it to Airbnb data.

For CV, a pretrained Deep Learning model called deepface was utilized.

- [NLP algorithms](https://github.com/jose-jaen/Airbnb/blob/main/Functions/nlp_functions.py)

- [Computer Vision algorithms](https://github.com/jose-jaen/Airbnb/blob/main/Functions/cv_functions.py)

# Machine Learning and Deep Learning

AI algorithms for predicting prices. In the ML part, Bayesian Ridge Regression, Elastic Net Regression, Random Forest, Bayesian Random Forest,
and XGBoost were used.

For DL, Artificial Neural Networks and Bayesian Neural Networks were built. 

Note that TPE algorithm (Bayesian Optimization) was used to select the best performing hyperparameters.

- [Machine Learning modeling](https://github.com/jose-jaen/Airbnb/blob/main/Functions/ml_models.py)

- [Deep Learning modeling](https://github.com/jose-jaen/Airbnb/blob/main/Functions/dl_models.py)

- [Bayesian Random Forest adaption](https://github.com/jose-jaen/Airbnb/blob/main/Functions/_forest.py)
