# Text pre-processing

# Import Gensim & NLTK

from gensim import corpora, models
from gensim.utils import simple_preprocess, lemmatize
from gensim.parsing.preprocessing import STOPWORDS as STOPWORDS

from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize

tokenizer = RegexpTokenizer(r'\w+')

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

from nltk.corpus import stopwords
import nltk


# create French stop word list
fr_stops = set(stopwords.words('french'))

# Add certain additional stop words
public_service_stops = '''public service canada work http 
https travail gcconnex url'''.split()

# Set up stemmer
stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def tokenize(text):
    return [lemmatize_stemming(token) for token in tokenizer.tokenize(str(text))
            if token not in STOPWORDS if token not in fr_stops
           if token not in public_service_stops if len(token) > 3]