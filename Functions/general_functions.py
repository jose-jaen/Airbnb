# Required packages
import io
import shap
import random
import requests
import datetime
import lime
import lime.lime_tabular
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from scipy import stats
from fitter import Fitter
from scipy.stats import pareto
from scipy.stats import median_test
from scipy.spatial.distance import cdist
from numpy.random import default_rng
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer 
from sklearn.inspection import PartialDependenceDisplay


def read_url(link):
    """ Creates a Pandas Dataframe from online data

    - Parameters:
        - link = Link to the zipped data

    - Output:
        - Pandas Dataframe
    """
    # Define URL and extract information
    response = requests.get(link)
    content = response.content

    # Convert into a Pandas Dataframe
    data = pd.read_csv(io.BytesIO(content), sep=',', compression='gzip')
    return data


def one_hot_encoder(word, data, text, aux):
    """ Creates binary variables based on the presence of a word in a comment

    - Parameters:
        - word = word to be identified in the text
        - data = dataframe to look in
        - text = column where the text is stored
        - aux = additional word to increase identification accuracy

    - Output:
        - Binary variable
    """
    data[word] = data[text].apply(lambda x: 1 if (word in x.lower() or aux in x.lower()) else 0)


def calculate_days(x, y):
    """ Retrieves number of days between given dates

    - Parameters:
        - x = Current date to serve as reference
        - y = Date from which to calculate days
    
    - Output:
        - Vector of elapsed days between dates
    """
    res = []
    y = y.replace(np.nan, x[0])
    for i in range(len(y)):
        d1 = datetime.datetime.strptime(y[i], '%Y-%m-%d')
        d2 = datetime.datetime.strptime(x[i], '%Y-%m-%d')
        delta = d2 - d1
        res.append(delta.days)
    return res


def new_id(data):
    """ Reset row index for a given dataset

    - Parameters:
        - data = Dataset whose indexes are to be reset

    - Output:
        - Newly indexed dataset
    """
    data = data.reset_index(drop=True)
    return data


def type_sort(data):
    """ Sorts a dataframe by variable type and counts feature classes

    - Parameters:
        -data = Unsorted Pandas Dataframe

    - Output:
        - X = Sorted Pandas Dataframe 
        - p1 = Number of quantitative variables
        - p2 = Number of binary variables
        - p3 = Number of multi variables
    """
    # Define data types
    binary = [i for i in data.columns if len(data[i].unique()) == 2]
    multi = [k for k in data.columns if data[k].dtype == 'category']
    quant = [j for j in data.columns if j not in binary and j not in multi]
    
    # Preparing output
    variables = quant + binary + multi
    X = pd.DataFrame(data.loc[:, variables])
    p1, p2, p3 = len(quant), len(binary), len(multi)
    return X, p1, p2, p3


def vgeom(D):
    """ Computes Geometric Variability

    - Parameters:
        - D = Squared Distance Matrix

    - Output:
        - Geometric Variability
    """
    n = D.shape[0]
    suma = np.sum(D, axis=1)
    return np.sum(suma)/2*n**2


def gower(X1, X2, X3):
    """ Computes statistical distance matrix for mixed data types

    - Parameters:
        - X1 = Continuous features matrix
        - X2 = Binary features matrix
        - X3 = Categorical features matrix

    - Output:
        - Gower Distance Matrix
    """
    # Distance Matrix for continuous variables
    S_inv = np.linalg.inv(np.cov(X1.T, bias=False))
    M = pairwise_distances(X1, metric='mahalanobis', n_jobs=-1, VI=S_inv)
    M2 = np.multiply(M, M)
    M /= vgeom(M2)

    # Distance Matrix for binary variables
    J = cdist(X2, X2, 'jaccard')
    J2 = np.multiply(J, J)
    J /= vgeom(J2)

    # Distance Matrix for categorical variables
    C = cdist(X3, X3, 'hamming')
    C2 = np.multiply(C, C)
    C /= vgeom(C2)

    # Gower Distance Matrix
    D = M + J + C
    return D


def KNN_Imputer(feature, target, k):
    """ Performs imputation based on K-NN Algorithm

    - Parameters:
        - feature = Data matrix with features
        - target = Vector containing target variable
        - k = Number of neighbors to consider

    - Output:
        - Non-missing feature data matrix
    """
    # Coerce into numeric data
    feature['target'] = target
    feature = feature.apply(pd.to_numeric, errors='coerce')

    # Dataset with no missing observations
    missing_cols = [i for i in feature.columns if feature[i].isna().sum() > 0]
    distance_data = feature.drop(missing_cols, axis=1)

    # Sort dataset by variable type
    distance_data, p1, p2, p3 = type_sort(distance_data)

    # Create three matrices based on data type
    p = p1 + p2 + p3
    X1 = distance_data.iloc[:, 0:p1] # continuous variables
    X2 = distance_data.iloc[:, p1:p1+p2] # binary variables
    X3 = distance_data.iloc[:, p1+p2:p] # categorical variables

    # Compute Gower Distance Matrix and replace diagonal by maximum value
    D = gower(X1, X2, X3)
    I = np.identity(D.shape[0])
    D = D + I*D.max()

    # Identify k-nearest neighbors based on Gower Distance
    KNN = np.argpartition(D, kth=k, axis=-1)
    
    # Replace missing values with the mean of neighbors
    for j in missing_cols:
        a = feature[j].isna()
        ids = [i for i in range(len(a)) if a[i] == True]
        for k in ids:
            closest = KNN[k, :5]
            feature.loc[k, j] = np.nanmean(feature.loc[closest, j])

    # In case some neighbors were also missing, perform iterative imputation
    if any(feature.isna().sum() > 0):
    imp = IterativeImputer(estimator=BayesianRidge(), 
                           max_iter=25, random_state=42)
        imp.fit(feature)
        feature = imp.transform(feature)
       
    feature = feature.drop('target', axis=1)
    return feature


def Iterative_Imputer(feature, target, model):
    """ Performs imputation based on Iterative Imputing Algorithm

    - Parameters:
        - feature = Data matrix with features
        - target = Vector containing target variable
        - model = Either Bayesian Ridge Regression or Random Forest

    - Output:
        - Non-missing feature data matrix
    """
    # Store column names
    cols = feature.columns.tolist()
    cols.append('target')

    # Coerce into numeric data
    feature['target'] = target
    feature = feature.apply(pd.to_numeric, errors='coerce')

    # Select model for Iterative Imputation Algorithm
    if model == 'bayesian':
        # Fit a Bayesian Ridge Regression model
        imp = IterativeImputer(estimator=BayesianRidge(), 
                               max_iter=25, random_state=42)
        imp.fit(feature)
        feature = imp.transform(feature)

        # Retrieve a pandas dataframe
        feature = pd.DataFrame(feature, columns=cols)
        feature = feature.drop('target', axis=1)

    else:
        # Fit a Random Forest model
        random_forest = RandomForestRegressor(max_depth=12, bootstrap=2,
                                              max_samples=0.5, n_jobs=-1, random_state=42)
        imp = IterativeImputer(estimator=random_forest, max_iter=25, random_state=42)
        imp.fit(feature)
        feature = imp.transform(feature)

        # Retrieve a pandas dataframe
        feature = pd.DataFrame(feature, columns=cols)
        feature = feature.drop('target', axis=1)
    return feature, target

def imputer_performance(data, imputer):
    """ Imputes data for algorithmic performance comparison
    
    - Parameters:
        - data = Original dataset to perform imputation
        - imputer = Data imputation algorithm
    
    - Output:
        - imputed_data = Final dataset with non-missing values
    """
    # Remove columns with missing values
    missing_cols = data[0].columns[data.isnull().any()]
    imputer_data = data[0].drop(missing_cols, axis=1)

    # Make a copy for later comparison
    original_data = imputer_data.copy()

    # Randomly introduce missing values
    rng = default_rng()
    random_cols = rng.choice(imputer_data.columns, size=10, replace=False)
    imputer_data.loc[3000:, random_cols] = np.nan

    # Impute data with different algorithms
    if imputer == 'knn':
        imputed_data = KNN_Imputer(imputer_data, data[1], k=6)
        
    elif imputer == 'bayes':
        imputed_data = Iterative_Imputer(imputer_data, data[1], model='bayesian')
        imputed_data = imputed_data[0]
        
    else:
        imputed_data = Iterative_Imputer(imputer_data, data[1], model='forest')
        imputed_data = imputed_data[0]
        
    performance = original_data.loc[3000:,random_cols].subtract(
        imputed_data.loc[3000:,random_cols]).sum()
    return performance


def XAI_SHAP(model, data, graph, obs):
    """ Computes SHAP values and represents XAI graphs

    - Parameters:
        - model = Machine Learning model to interpret
        - data = Data used to make explanations
        - graph = Global or local interpretation
        - obs = Index of data instance to explain

    - Output:
        - XAI graphs and SHAP values
    """
    # Print JavaScript visualizations
    shap.initjs()

    # Create object to calculate SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    
    if graph == 'global':
        # Global Interpretability (feature importance)
        shap.plots.bar(shap_values, max_display=20)
        
        # Global Interpretability (impact on target variable)
        shap.summary_plot(shap_values, data, max_display=20)
        
    else:
        # Local Interpretability (coefficients)
        shap.plots.bar(shap_values[obs], max_display=20)
        shap.plots.waterfall(shap_values[obs], max_display=20)

        # Local Interpretability (force plots)
        shap.plots.force(shap_values[obs])
    return shap_values


def XAI_LIME(model, data, obs):
    """ Represents LIME plot

    - Parameters:
        - model = Machine Learning model to interpret
        - data = Data used to make explanations
        - observation = Number of listing to explain

    - Output:
        - LIME plot
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(data.values,
                                                       feature_names=data.columns.values.tolist(),
                                                       verbose=True, mode='regression')

    # Explain the selected listing observation
    exp = explainer.explain_instance(data.values[obs], model.predict, num_features=20)

    # Show the predictions
    return exp.show_in_notebook(show_table=True)


def XAI_PDP_ICE(model, data, var1, var2, ice: bool):
    """ Represents PDP and ICE plots

    - Parameters:
        - model = Machine Learning model to interpret
        - data = Data used to make explanations
        - var1 = Index of first feature to explain
        - var2 = Index of second feature to explain
        - ICE = Whether to include ICE plots

    - Output:
        - PDP and ICE plots
    """
    # Create vector for feature comparison
    features = [var1, var2, (var1, var2)]

    # Check if user asks only for PDP plots
    if ice:
        PartialDependenceDisplay.from_estimator(model, data,
                                                features, kind=['both', 'both', 'average'])
        
    else:
        PartialDependenceDisplay.from_estimator(model, data, features)
