# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 15:31:02 2018

@author: Wolf
"""
# =============================================================================
#               NATURAL LANGUAGE PROCESSING
# =============================================================================
# Importi vseh potrebnih knjižnic
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
try:
    from StringIO import StringIO
    from BytesIO import BytesIO
except ImportError:
    from io import StringIO
    from io import BytesIO

stemmer = PorterStemmer() 
lemmatizer = WordNetLemmatizer() 

# Če je problem z tokeniziranjem zaradi errorja "None type" - probaj z custom_tokenize najprej
def custom_tokenize(text):
    if not text:
        print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return word_tokenize(text)

# =============================================================================
#
#  PROTOTYPING FUNCTION
#  params: dataframe, feature to qualify
#  example: fq_barplot(df2, 'qm_per_comment_additional')
#
# =============================================================================
def fq_barplot(df, feature):
    dffunc = pd.concat([df, pd.DataFrame({'ax{}'.format(i): df['type'].astype(str).str[i] for i in range(4)})], axis=1)
    fig, ax = plt.subplots(2,2,figsize=(10,8))
    ax = ax.flatten()
    fig.suptitle('Kvaliteta vektorja = '+feature)
    sns.set(style="white", color_codes=True)
    for i in range(4):
        sns.barplot(x='ax{}'.format(i), y=feature, data=dffunc, ax=ax[i], ci='sd')

#fq_barplot(df2, 'qm_per_comment')

# PREPOČASN
# =============================================================================
# def fq_swarmplot(df, feature):
#     dffunc = pd.concat([df, pd.DataFrame({'ax{}'.format(i): df['type'].astype(str).str[i] for i in range(4)})], axis=1)
#     fig, ax = plt.subplots(2,2,figsize=(10,8))
#     ax = ax.flatten()
#     fig.suptitle('Kvaliteta vektorja = '+feature)
#     sns.set(style="white", color_codes=True)
#     for i in range(4):
#         sns.swarmplot(x='ax{}'.format(i), y=feature, data=dffunc, ax=ax[i])
# 
# =============================================================================
#fq_swarmplot(df2, 'qm_per_comment')

# =============================================================================
#           PRIPRAVA PODATKOV oz. KOMENTARJEV
# TODO: Popravit je potrebno indeksiranje na koncu, ker izgubimo komentarje z linki.
# =============================================================================
# Če pride do korupcije dataseta, poženi še enkrat    
# del df
# del df_melt
# =============================================================================

# =============================================================================
# df=pd.read_csv('data/mbti_1.csv')
# df.info()
# 
# # Imamo 50 komentarjev za vsakega uporabnika, zato naredimo tudi 50 stolpcev za vsak komentar.
# columnlist=[str(i) for i in range(50)]
# # Splitamo komentarje v vsak stolpec posebej in droppamo originalne "posts" - jih ne potrebujemo
# df[columnlist] = df['posts'].str.split(r'\|\|\|',n=49, expand=True)
# df.drop(['posts'],inplace=True, axis=1)
# 
# #Vse komentarje damo v nov stolpec "post"
# df_melt = pd.melt(df, id_vars=['type'], value_vars=columnlist, value_name='post')
# df_melt.drop(['variable'], inplace=True, axis=1)
# #Stripa vse "'" na začetku nekaterih komentarjev
# df_melt['post'] = df_melt['post'].str.lstrip("'")
# 
# # Moramo filtrirat ven komentarje z linki, drugače je problem z dolžino komentarjev (TypeError: object of type 'NoneType' has no len())
# # To je začasno - bomo probali nardit par featurjev in domnev samo preko stila pisanja, sentimenta, itd.
# # V nadaljevanju porihtamo, da bojo še linki delali in se bomo osredotočili še na tiste featurje
# df_melt = df_melt[df_melt['post'].str.contains('http') == False]
# # Ostane nam = 396313 rows
# 
# #Null vrednosti - mogoče uporabno za naprej
# #nulls = sum(df_melt['post'].isnull() == True)
# #Dolžina komentarja
# df_melt['post_length'] = df_melt['post'].apply(lambda x: len(x))
# df_melt['post'][1]
# =============================================================================
# Backup
# df_melt.to_csv('mbti_consolidated.csv',index=False)

# Inicializacija ali popravek, če si kej zajebal :D
# del df2
# df2 = df_melt

df=pd.read_csv('data/mbti_1.csv')
df.info()

posts_df = []
for uid in df.index:
    psts = df.loc[uid, 'posts'].strip("'").split(r'|||')
    psts = pd.DataFrame({'type': df.loc[uid, 'type'], 'post':pd.Series(psts)})
    psts['post_length'] = psts['post'].apply(lambda x: len(x))
    psts['uid'] = uid
    posts_df.append(psts)

posts_df = pd.concat(posts_df).reset_index(drop=False)
posts_df.to_csv('mbti_consolidated2.csv',index=False)

test = pd.read_csv('mbti_consolidated2.csv', encoding='latin1')


# =============================================================================
# =============================================================================
#           PRIPRAVA STEM in LEM DATAFRAME - for easy use
#   Attention! 'post_length' is the length of the post before data was wrangled and removed special characters, etc.
#   
#   Popravit je bilo treba ročno empty values v csv-jih, da obdržimo enake 
#   indekse in je pravi komentar na pravem indeksu in tipu. 
#   !!!! Prazni komentarji so spremenjeni v 'NaN' in prebrani inicialno v df KOT STRING !!!
#
#
#       USE dflem - Dataframe with lemmatized posts
#       USE dfstem - Dataframe with stemmed posts
# =============================================================================
dfcon = pd.read_csv("mbti_consolidated2.csv", encoding="latin1") #Z http linki.
dfcon.dropna(axis=0, inplace=True)

# =============================================================================
#               STEMMING and LEMMATIZING moved to -> stem_lemmat.py
# =============================================================================
dflem = pd.read_csv("mbti_corpus_lemmatized.csv", header=None)
dfstem = pd.read_csv("mbti_corpus_stemmed.csv", header=None)
dflem.columns = ['post']
dfstem.columns = ['post']

x2 = dfcon.drop('post', axis=1, inplace=False)
x2 = pd.merge(x2, dflem, how='inner', left_index=True, right_index=True)
x2 = x2[['type', 'post', 'post_length']]
dflem = x2
del x2

x3 = dfcon.drop('post', axis=1, inplace=False)
x3 = pd.merge(x3, dfstem, how='inner', left_index=True, right_index=True)
x3 = x3[['type', 'post', 'post_length']]
dfstem = x3
del x3
# =============================================================================
#dflem['denied_words_per_comment'] = dflem['post'].apply(lambda x: str(x).count('cannot') + str(x).count('can not'))

# =============================================================================
#                       NLTK knjižnica - delo
# =============================================================================
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *


#n_instances = 100
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')] #[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')] #[:n_instances]]
len(subj_docs), len(obj_docs)

train_subj_docs = subj_docs[:4000]
test_subj_docs = subj_docs[4000:5000]
train_obj_docs = obj_docs[:4000]
test_obj_docs = obj_docs[4000:5000]
training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs

sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])

unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)

for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))
    
    

#dfcon = dfcon.iloc[:10]
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import tokenize

setnan = SentimentAnalyzer()
sentinan = SentimentIntensityAnalyzer()

#content = tokenize.sent_tokenize(str(dfcon.post))
sentences = []
for i, row in dfcon.iterrows():
#    ss = setnan.evaluate(row['post'])
    ss = sentinan.polarity_scores(str(row['post']))
    ssarray = [ss['neg'],ss['neu'],ss['pos'], ss['compound']]
    sentences.append(ssarray)
    
polar = pd.DataFrame(data=booksent, columns=['neg', 'neu', 'pos', 'compound'])
polar.reset_index(drop=True,inplace=True)
dfpolar = pd.merge(dfcon, polar, left_index=True, right_index=True)    

dfcon = dfcon[['index', 'type', 'post', 'uid', 'post_length']]
dfpolar[[]]

fq_barplot(dfpolar, 'neg')

print(polar.shape)
print(dfcon.shape)

print(polar.head(10))
print(dfcon.head(10))

print(dfcon[pd.isnull(dfcon).any(axis=1)])
print(dfcon['post'].isnull() == True)


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
dataraw = pd.read_csv('mbti_consolidated2.csv', encoding='latin1') #text in column 1, classifier in column 2.
#intro_extra = pd.read_csv('data/mbti_1.csv')

data = dataraw.copy()


def evalSGD():
    j=0 #for iterating
    k=1 #for iterating through axioms
    _score = []
    for r in range(4):
        for i in range(len(data)):
            data['type'].values[i] = dataraw['type'].values[i][j:k]
        
        temp = data.copy()
        temp = temp[['index', 'type', 'post', 'uid']]
        temp.dropna(axis=0, inplace=True)
        
        numpy_array = temp.as_matrix()
        Y = numpy_array[:,1]
        X = numpy_array[:,2]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=42)
        ## BAYES
        text_clf_sgd = Pipeline([('vect', CountVectorizer(stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, max_iter=10, random_state=42)),
        ])
        text_clf_sgd = text_clf_sgd.fit(X_train,Y_train)
        predicted_svm = text_clf_sgd.predict(X_test)
        
        j += 1
        k += 1
        _score.append(np.mean(predicted_svm == Y_test))

    ax1 = ax2 = ax3 = ax4 = 0
    ax1 = _score[0]
    ax2 = _score[1]
    ax3 = _score[2]
    ax4 = _score[3]
    print('     Stochastic Gradient Descent Classifier  (Ocene napovedi modela)   ')
    print('-----------------------------------------------------------------------')
    print('Introvert (I) - Extrovert  (E):               {0}'.format(ax1))
    print('Intuition (N) - Sensing    (S):               {0}'.format(ax2))
    print('Thinking  (T) - Feeling    (F):               {0}'.format(ax3))
    print('Judging   (J) - Perceiving (P):               {0}'.format(ax4))

evalSGD()

