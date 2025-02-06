import pandas as pd
from nltk.stem import PorterStemmer
import re
import torch
import numpy as np
import nltk
import string

nltk.download('punkt_tab')

df = pd.read_csv("C:\Python Code\Real Estate\data.csv")

text = "Brother's suck. I dislike them most of the time. In fact, while running once, they tripped me on purpose's acres real estate central bonus room upper floors as they include"

stemmer = PorterStemmer()
def tokenize_and_stem(text):
    tokens = nltk.tokenize.word_tokenize(text)
    # strip out punctuation and make lowercase
    tokens = [token.lower().strip(string.punctuation)
              for token in tokens if token.isalnum()]

    # now stem the tokens
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

print(tokenize_and_stem(text))


