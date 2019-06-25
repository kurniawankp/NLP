import pandas as pd 
import numpy as np 
import nltk
from nltk.corpus import stopwords
from pywsd.utils import lemmatize_sentence
import re,string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class model():

    def __init__(self):
        self.negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
        self.neg_pattern = re.compile(r'\b(' + '|'.join(self.negations_dic.keys()) + r')\b')
        

    def preprocessing(self,sentence):
        
        clean_text = []
        sentence = sentence.lower()
        cleanr = re.compile('<.*?>')
        sentence = re.sub(cleanr, ' ', sentence) #menghilangkan HTML tag
        sentence = re.sub(r'[?|!|\'|"|#]',r'', sentence) 
        sentence =  re.sub(r'[.|,|)|(|\|/]',r' ',sentence)#menghilangkan punctuation
        sentence = self.neg_pattern.sub(lambda x: self.negations_dic[x.group()], sentence) #menyimpan kata negasi agar tidak hilang
        
        for word in sentence.split():
            if word not in stopwords.words('english'):
                word = lemmatize_sentence(word)
                word = word[0]
                clean_text.append(word)
                
        return (" ".join(clean_text))

    def train(self)


        
        
