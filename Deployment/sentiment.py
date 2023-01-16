from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import logging
from scipy.special import softmax

logging.set_verbosity_error()

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(' '):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return ' '.join(new_text)

def load_model():
    model = AutoModelForSequenceClassification.from_pretrained('./Airbnb/Deplyoment/')
    return model
