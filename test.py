import pandas as pd 
import numpy as np 
import nltk
from nltk.corpus import stopwords
from pywsd.utils import lemmatize_sentence
import re,string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def clean_sentence(sentence):
    negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
    clean_text = []
    sentence = sentence.lower()
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence) #menghilangkan HTML tag
    sentence = re.sub(r'[?|!|\'|"|#]',r'', sentence) 
    sentence =  re.sub(r'[.|,|)|(|\|/]',r' ',sentence)#menghilangkan punctuation
    sentence = neg_pattern.sub(lambda x: negations_dic[x.group()], sentence) #menyimpan kata negasi agar tidak hilang
    
    for word in sentence.split():
        if word not in stopwords.words('english'):
            word = lemmatize_sentence(word)
            word = word[0]
            clean_text.append(word)
            
    return (" ".join(clean_text))

def feature(dataframe):
    df = dataframe
    X = df.iloc[:,0]
    y = df.iloc[:,1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    cv = TfidfVectorizer(ngram_range=(1,3))
    x_traincv = cv.fit_transform(X_train)
    y_train = y_train.astype('int')
    x_test = 



data_dir = '/Users/hishshahghassani/Desktop/kp/data set/NLP/all/train.tsv'
df = pd.read_csv(data_dir,sep='\t')

print(df.iloc[:,3])
'''
clean = df['Phrase'].apply(clean_sentence)
sentiment = df['Sentiment']
train = pd.DataFrame([clean,sentiment],columns=['phrase','sentiment'])
train.dropna(axis=0,inplace=True)
train.reset_index(drop=True,inplace=True)

X = train['phrase']
y = train['sentiment']
'''


