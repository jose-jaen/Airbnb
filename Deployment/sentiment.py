from scipy.special import softmax
from transformers import (
    AutoModelForSequenceClassification,
    TFAutoModelForSequenceClassification,
    AutoTokenizer, 
    AutoConfig, 
    logging
)

logging.set_verbosity_error()

# Preprocess text
def preprocess(text):
    new_text = []
    for t in text.split(' '):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return ' '.join(new_text)

# Load transformer model
def load_model():
    PATH = f'cardiffnlp/twitter-roberta-base-sentiment-latest'
    tokenizer = AutoTokenizer.from_pretrained(PATH)
    config = AutoConfig.from_pretrained(PATH)
    model = AutoModelForSequenceClassification.from_pretrained(PATH)
    return model, tokenizer, config
