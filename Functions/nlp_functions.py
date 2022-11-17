# Required packages
import re
import string
import lxml
from langid import classify
from bs4 import BeautifulSoup
import nltk.corpus
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# NLP functions for data text preparation and Machine Learning modeling

def only_latin(x):
    """ Removes non-latin characters

    - Parameters:
        - x = Dataset with non-latin characters

    - Output:
        - Cleaned dataset
    """
    x = x.str.encode('ascii', 'ignore')
    x = x.astype('|S')
    x = [x[i].decode('utf-8') for i in range(len(x))]
    return x


def lang_identifier(x, text):
    """ Eliminates non-English text from an array

    - Parameters:
        - x = Dataset with information
        - text = Column with comments

    - Output:
        - English comments
    """
    non_english = [i for i in range(len(x[text])) if classify(x[text][i])[0] != 'en']
    data = x.drop(labels=non_english, axis=0)
    return data


def clean_text(text):
    """ Removes symbols, stopwords and simplifies text

    - Parameters:
        - text = Vector of comments to clean

    - Output:
        - Cleansed comments
    """
    # Remove stopwords
    stop = stopwords.words('english')
    stemmer = PorterStemmer()
    
    # Eliminate punctuation
    exclude = set(',.:;')
    text = ''.join([(ch if ch not in exclude else ' ') for ch in text])  
    
    # Regex to ignore urls and special symbols
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    
    # Replace certain words
    text = text.replace('&', 'and')
    text = ' '.join([word for word in text.split() if word not in stop])
    text = BeautifulSoup(text, 'lxml').text
    
    # Force to lowcase
    text = text.lower()
    
    # Delete some characters
    text = text.replace('\r<br/>', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('/', ' ')
    text = text.replace('...', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('(', '')
    text = text.replace(')', '')
    return text


def sentiment_vader(sentence):
    """ Performs Sentiment Analysis with a pre-trained NLP model.
    Some rules are manually defined so as to increase accuracy

    - Parameters:
        - sentence = Comment to analyze

    - Output:
        - negative = Negative sentiment score
        - neutral = Neutral sentiment score
        - positive = Positive sentiment score
        - compound = Compound sentiment score
        - overall_sentiment = Average sentiment
    """
    # Create a SentimentIntensityAnalyzer object
    SIA = SentimentIntensityAnalyzer()

    # Define custom verbal rules
    SIA.lexicon.update({'cancel': -20})
    SIA.lexicon.update({'canceled': -20})
    SIA.lexicon.update({'canceling': -20})
    SIA.lexicon.update({'backpain': -5})
    SIA.lexicon.update({'bad': -10})
    SIA.lexicon.update({'spiderwebs': -50})
    SIA.lexicon.update({'odor': -30})
    SIA.lexicon.update({'freaked': -30})
    SIA.lexicon.update({'musty': -50})
    SIA.lexicon.update({'toxic': -5})
    SIA.lexicon.update({'sticky': -15})
    SIA.lexicon.update({'ugly': -15})
    SIA.lexicon.update({'bedbugs': -60})
    SIA.lexicon.update({'bugs': -20})
    SIA.lexicon.update({'rude': -30})
    SIA.lexicon.update({'aggressive': -30})
    SIA.lexicon.update({'scary': -15})
    SIA.lexicon.update({'cozy': +10})
    SIA.lexicon.update({'great': +20})
    SIA.lexicon.update({'cosy': +10})
    SIA.lexicon.update({'smoothly': +30})
    SIA.lexicon.update({'worst': -10})
    SIA.lexicon.update({'convenient': +40})
    SIA.lexicon.update({'worse': -10})
    SIA.lexicon.update({'exciting': +60})
    SIA.lexicon.update({'notch': +30})
    SIA.lexicon.update({'superhost': +30})
    SIA.lexicon.update({'disappointing': -10})
    SIA.lexicon.update({'horrible': -30})
    SIA.lexicon.update({'dirty': -15})
    SIA.lexicon.update({'dirt': -15})
    SIA.lexicon.update({'stain': -20})
    SIA.lexicon.update({'filthy': -30})
    SIA.lexicon.update({'unreliable': -15})
    SIA.lexicon.update({'meh': -5})
    SIA.lexicon.update({'spacious': +10})
    SIA.lexicon.update({'lovely': +20})
    SIA.lexicon.update({'infested': -15})
    SIA.lexicon.update({'broke': -10})
    SIA.lexicon.update({'broken': -15})
    SIA.lexicon.update({'awake': -20})
    SIA.lexicon.update({'difficult': -20})
    SIA.lexicon.update({'1010': +10})
    
    # Create dictionary with possible outcomes
    sentiment_dict = SIA.polarity_scores(sentence)
    negative, neutral = sentiment_dict['neg'], sentiment_dict['neu']
    positive, compound = sentiment_dict['pos'], sentiment_dict['compound']

    # Vector with sentiments
    sent_vec = [negative, neutral, positive]

    # Thresholds for sentiment analysis
    # Positive sentiment for high scoring comments
    if sentiment_dict['compound'] >= 0.05:
        overall_sentiment = 1

    # Negative sentiment for lowest scoring comments
    elif sentiment_dict['compound'] <= - 0.05 and sent_vec.index(max(sent_vec[0:3])) == 0:
        overall_sentiment = -1

    # Neutral sentiment for remaining comments
    else:
        overall_sentiment = 0

    return overall_sentiment


def sentiment(x):
    """ Retrieves encoded sentiment from NLP model

    - Parameters:
        - x = Text data array
    
    - Output:
        - Encoded Sentiment Analysis result
    """
    sent = [sentiment_vader(clean_text(x[i])) for i in range(len(x))]
    return sent
