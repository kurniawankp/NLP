import pandas as pd
df = pd.read_csv('/Users/hishshahghassani/Desktop/data set/NLP/all/train.tsv', sep = '\t')
#data didownload di https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data
import re, string
import nltk
from nltk.corpus import stopwords
from pywsd.utils import lemmatize_sentence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
    
def clean_text(text):
    clean_text = []
    text = text.lower()
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, ' ', text) #menghilangkan HTML tag
    text = re.sub(r'[?|!|\'|"|#]',r'', text) 
    text =  re.sub(r'[.|,|)|(|\|/]',r' ',text)#menghilangkan punctuation
    text = neg_pattern.sub(lambda x: negations_dic[x.group()], text) #menyimpan kata negasi agar tidak hilang
    
    for word in text.split():
        if word not in stopwords.words('english'):
            word = lemmatize_sentence(word)
            word = word[0]
            clean_text.append(word)
            
    return (" ".join(clean_text))
print("tahap 1 berhasil")

#membersihkan data text
clean_phrase = []
for x in df['Phrase']:
    clean_phrase.append(clean_text(x))
print("tahap 2 berhasil")

#membuat data frame untuk data yang telah di processing
clean_df = pd.DataFrame(clean_phrase,columns=['Phrase'])

#membuat target data yang telah di process sesuai data set
clean_df['target'] = df.Sentiment

#membuat data frame data yang telah di process ke dalam csv
clean_df.to_csv('clean_phrase.csv',encoding='utf-8')

csv = 'clean_phrase.csv'
my_df = pd.read_csv(csv,index_col=0)

#membuang data yang isinya Nan
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)

print("tahap 3 berhasil")

df_x = my_df['Phrase']
df_y = my_df['target']

#membuat fitur dengan Tfidf


x_train, y_train = df_x, df_y
cv1 = TfidfVectorizer(ngram_range=(1,3))
x_traincv = cv1.fit_transform(x_train)
y_train = y_train.astype('int')

print("tahap 4 berhasil")

#mengtraining data
model = LogisticRegression()
model.fit(x_traincv, y_train)
print("tahap 5 berhasil")

#menyimpan model
import pickle
filename = 'model_sentiment.sav'
pickle.dump(model, open(filename, 'wb'))

print("tahap 6 berhasil")

