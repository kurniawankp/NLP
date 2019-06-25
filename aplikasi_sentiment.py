from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import pickle
import pandas as pd
import pickle
import re, string
import nltk
from nltk.corpus import stopwords
from pywsd.utils import lemmatize_sentence
from sklearn.feature_extraction.text import TfidfVectorizer
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
#membuat fungsi untuk memproses data text
def clean_text(text):
    clean_text = []
    text = text.lower()
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, ' ', text) #remove HTML tag
    text = re.sub(r'[?|!|\'|"|#]',r'', text) 
    text =  re.sub(r'[.|,|)|(|\|/]',r' ',text)#remove punctuation
    text = neg_pattern.sub(lambda x: negations_dic[x.group()], text)
    
    for word in text.split():
        if word not in stopwords.words('english'):
            word = lemmatize_sentence(word)
            word = word[0]
            clean_text.append(word)
            
    return (" ".join(clean_text))

#mengimport data yang telah diproses untuk di fit menggunakan TfidfVectorizer
csv = 'clean_phrase.csv'
my_df = pd.read_csv(csv,index_col=0)
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
cv1 = TfidfVectorizer(ngram_range=(1,3))
x = cv1.fit_transform(my_df.Phrase)


class MyApp:

    def __init__(self, master):
        
        master.title('Aplikasi Sentiment Analysis')
        master.resizable(False, False)
        master.configure(background = '#0000b9')

        self.style = ttk.Style()
        self.style.configure('TFrame', background = 'white')
        self.style.configure('TButton', background = 'white')
        self.style.configure('TLabel', background = 'white', font = ('Arial', 11))
        self.style.configure('Header.TLabel', font = ('Arial', 18, 'bold'))      

        self.frame_header = ttk.Frame(master)
        self.frame_header.pack()

        ttk.Label(self.frame_header, text = 'Sentiment Analyzer', style = 'Header.TLabel').grid(row = 0, column = 1)
        ttk.Label(self.frame_header, wraplength = 200,
                  text = ("Masukkan sebuah kalimat pada kotak yang tersedia.  "
                          "Klik tombol Analyze untuk menentukan klasifikasi kalimat")).grid(row = 1, column = 1)
        
        self.frame_content = ttk.Frame(master)
        self.frame_content.pack()

        ttk.Label(self.frame_content, text = 'Kalimat:').grid(row = 0, column = 0, padx = 5, sticky = 'sw')
        
        self.entry_kalimat = ttk.Entry(self.frame_content, width = 24, font = ('Arial', 10))
        
        
        self.entry_kalimat.grid(row = 1,rowspan=2, column = 0, columnspan=2, padx = 5)
        
        
        ttk.Button(self.frame_content, text = 'Analyze', 
                   command = self.submit).grid(row = 4, column = 0, padx = 5, pady = 5, sticky = 'e')
        ttk.Button(self.frame_content, text = 'Clear',
                   command = self.clear).grid(row = 4, column = 1, padx = 5, pady = 5, sticky = 'w')

    
    def submit(self):
        #testing klasifikasi dengan kalimat custom
        custom = (self.entry_kalimat.get())
        custom_clean = clean_text(custom)
        custom_list = [custom_clean]
        custom_test = cv1.transform(custom_list)
        filename = 'model_sentiment.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        prediksi = loaded_model.predict(custom_test)
        if prediksi[0] == 0:
            pred_val = 'negatif'
        if prediksi[0] == 1:
            pred_val = 'semi negatif'
        if prediksi[0] == 2:
            pred_val = 'neutral'
        if prediksi[0] == 3:
            pred_val = 'semi positif'
        if prediksi[0] == 4:
            pred_val = 'positif'
        
        print('Kalimat: {}'.format(self.entry_kalimat.get()))
        
        self.clear()
        messagebox.showinfo(title = 'Aplikasi Sentiment Analysis', message = 'Prediksi kelas: {}'.format(pred_val))
    
    def clear(self):
        self.entry_kalimat.delete(0, 'end')
        
       
def main():            
    
    root = Tk()
    myApp = MyApp(root)
    root.mainloop()
    
if __name__ == "__main__": main()