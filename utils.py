import nltk
from string import punctuation, digits, ascii_lowercase
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re

nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

from nltk.util import ngrams

nltk.download('punkt')
from nltk.tokenize import word_tokenize

nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag

from rake_nltk import Rake

rem_punct = lambda text: ''.join([char.lower() for char in text if char not in punctuation])
rem_nonletter = lambda text: ''.join([char.lower() for char in text if char.lower() in ascii_lowercase + digits + ' '])
rem_stop = lambda text: ' '.join([word for word in text.split(' ') if word not in stop])

clean = lambda text, strict=False: word_tokenize(rem_stop(rem_punct(text))) if not strict else word_tokenize(rem_stop(rem_nonletter(text)))

def run_rake(text, min_length=2, max_length=4):
    rake = Rake(min_length=min_length, max_length=max_length, stopwords=stop, punctuations=punctuation)
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

def get_pos(text, pos='noun'):
    tags = pos_tag(word_tokenize(text))
    if pos == 'noun':
        words = [word for word, pos in tags if pos in ['NN', 'NNP']]
    return words

