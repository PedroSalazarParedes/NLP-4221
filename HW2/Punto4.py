import os
import os.path as op
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(op.dirname(op.abspath(__file__)))
workspace = op.dirname(op.abspath(__file__))

import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

en_stopwords = set((stopwords.words('english')))
es_stopwords = set((stopwords.words('spanish')))
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
from nltk.stem import SnowballStemmer

es_stemmer = SnowballStemmer('spanish')
import math
import re

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing

from utils import *
from models import *


df = pd.read_csv(f'{workspace}/fnmourning.csv')

df['text'] = df['text'].apply(clean_up_pipeline)
df['text'] = df['text'].apply(word_tokenize)

df_en = df.loc[df['lang']=='en']
df_es = df.loc[df['lang']=='es']

df_en['text'] = df_en['text'].apply(lambda text: remove_stopwords(text,en_stopwords))
df_es['text'] = df_es['text'].apply(lambda text: remove_stopwords(text,es_stopwords))

df_en['text'] = df_en['text'].apply(lambda text: stem_words(text,'en'))
df_es['text'] = df_es['text'].apply(lambda text: stem_words(text,'es'))

data_en = list(zip(df_en.index,df_en['text']))
data_es = list(zip(df_es.index,df_es['text']))

list_tweets_processed = []
for id, text in data_en:
    list_tweets_processed.append(text)

list_tweets_processed_es = []
for id, text in data_es:
    list_tweets_processed_es.append(text)

vocabulary_en = create_term_index(list_tweets_processed)
vocabulary_es = create_term_index(list_tweets_processed_es)


def get_nb_results(list_tweets,df,vocabulary):
    mlb = MultiLabelBinarizer()
    mlb.fit(list_tweets)

    df_en_m = df.loc[df['tag']=='mourning']
    df_en_nom = df.loc[df['tag']=='no mourning']

    data_en_mourn = list(zip(df_en_m.index,df_en_m['text']))

    list_tweets_processed_mourn = []
    for id, text in data_en_mourn:
        list_tweets_processed_mourn.append(text)

    X_en_mourn = mlb.transform(list_tweets_processed_mourn)
    english_mourn = pd.DataFrame(X_en_mourn, columns = mlb.classes_).T

    data_en_nomourn = list(zip(df_en_nom.index,df_en_nom['text']))

    list_tweets_processed_nomourn = []
    for id, text in data_en_nomourn:
        list_tweets_processed_nomourn.append(text)

    X_en_nomourn = mlb.transform(list_tweets_processed_nomourn)
    english_nomourn = pd.DataFrame(X_en_nomourn, columns = mlb.classes_).T

    pmourn = np.log(sum(df['tag']=='mourning')/len(df['tag']))
    pnomourn = np.log(sum(df['tag']=='no mourning')/len(df['tag']))

    vocabulary_en_mourn = create_term_index(list_tweets_processed_mourn)
    vocabulary_en_nomourn = create_term_index(list_tweets_processed_nomourn)

    wic_en_mourn = (english_mourn.sum(axis=1)+1)/(len(vocabulary_en_mourn)+len(vocabulary))
    wic_en_nomourn = (english_nomourn.sum(axis=1)+1)/(len(vocabulary_en_nomourn)+len(vocabulary))

    nb_en = pd.DataFrame(dict(wic_en_mourn = wic_en_mourn, wic_en_nomourn = wic_en_nomourn)).reset_index()
    nb_en['wic_en_mourn'] = np.log(nb_en['wic_en_mourn']) 
    nb_en['wic_en_nomourn'] = np.log(nb_en['wic_en_nomourn'])
    nb_en['nb_mourn'] = pmourn + nb_en['wic_en_mourn']
    nb_en['nb_nomourn'] = pnomourn + nb_en['wic_en_nomourn']

    nb_en['nb_class'] = np.where(nb_en['nb_mourn']>nb_en['nb_nomourn'],'mourning','no mourning')

    mourn_nb = nb_en[nb_en['nb_class'] == 'mourning'].loc[:,['index','nb_mourn']].sort_values(by=['nb_mourn'],ascending=False).set_index('index')
    nomourn_nb = nb_en[nb_en['nb_class'] == 'no mourning'].loc[:,['index','nb_nomourn']].sort_values(by=['nb_nomourn'],ascending=False).set_index('index')
    return mourn_nb, nomourn_nb, nb_en

mourn_nb_en, nomourn_nb_en, nb_en = get_nb_results(list_tweets_processed,df_en,vocabulary_en)
mourn_nb_es, nomourn_nb_es, nb_es = get_nb_results(list_tweets_processed_es,df_es,vocabulary_es)

scaler = preprocessing.MinMaxScaler()
es_scaler = preprocessing.MinMaxScaler()

mourn_nb_en['score'] = scaler.fit_transform(mourn_nb_en)
nomourn_nb_en['score'] = scaler.transform(nomourn_nb_en)

mourn_nb_es['score'] = es_scaler.fit_transform(mourn_nb_es)
nomourn_nb_es['score'] = es_scaler.transform(nomourn_nb_es)

plt.figure()
words = get_topwords(mourn_nb_en,50)
words.plot(kind='bar')
plt.show()

dict_mourn_en = get_dictionary(mourn_nb_en)
dict_nomourn_en = get_dictionary(nomourn_nb_en)
dict_mourn_es = get_dictionary(mourn_nb_es)
dict_nomourn_es = get_dictionary(nomourn_nb_es)

df_en = create_features(df_en,dict_mourn_en,dict_mourn_en)
df_en = create_features(df_es,dict_mourn_es,dict_mourn_es)

cols = ['NUM_TOKENS','COUNT_MOURN','SUM_MOURN','COUNT_NOMOURN','SUM_NOMOURN','MOURN_PERC','NOMOURN_PERC','Y']

X, y = process_df(df_en,cols)
X_es, y_es = process_df(df_es,cols)

result = get_results(X,y,23,0.25)
result_es = get_results(X_es,y_es,23,0.25)

_, _, lr = train_model(X,y,23,0.25,'LR')
lr_coef = lr.coef_
lr_int = lr.intercept_

perm_importance = get_rfimportance(X,y,23,0.25)
#sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(cols[:-1], perm_importance.importances_mean)
plt.xlabel("Permutation Importance")


test = 'test'
