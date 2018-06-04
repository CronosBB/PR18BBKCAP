# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:39:56 2018

@author: CronosBB
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import nltk

#nltk.download()

data = pd.read_csv('data/mbti_1.csv')
data.head()

## Splitamo vsak komentar za varianco oz. odstopanje
def var_row(row):
    l = []
    for i in row.split('|||'):
        l.append(len(i.split()))
    return np.var(l)

##
df = data    
df['words_per_person'] = data['posts'].apply(lambda x: len(x.split()))
df['number_comments_per_person'] = data['posts'].apply(lambda x: len(x.split('|||')))
#df[['words_per_person','number_comments_per_person']]
#df['words_per_comment_splitted'] = df['posts'].apply(lambda x: x.split())
df['variance_of_word_counts'] = data['posts'].apply(lambda x: var_row(x))
#df.head(20)

#df.describe

## Ustvari Series komentarjev
dfstr = pd.DataFrame(data=data['posts'])
##dftemp2 = re.sub(r'\|||', '',  dftemp)
dftemp = dfstr['posts'].str.replace('\n', '')
dftemp = dfstr['posts'].apply(lambda x: x.split('|||')).apply(pd.Series)
#dftemp.index = dftemp.set_index(df.index).index
dftemp.index = dfstr.set_index(df.index).index
postsSeries = dftemp.stack()
postsSeries.head(60)
## In imamo pripravljeno Series

## Mogoče bomo potrebovali še DataFrame
## indexpc = index na person/comment is Series
#dfposts = pd.DataFrame({'indexpc':postsSeries.index, 'comment':postsSeries.values})
#dfposts.set_index(keys=dfposts['indexpc'], inplace=True)
#del dfposts['indexpc']
#dfposts.head(20)


#### POZAB
#columns = ['type', 'posts', 'comment']
#dftemp2 = pd.DataFrame(data=df, index=df.index, columns=columns)
#dfcomments2 = pd.concat([pd.Series(row['comment'], row['posts'].split('|||')) for _, row in dftemp2.iterrows()]).to_frame().stack().reset_index()
####


####### EXTRACTING EACH COMMENT TO IT'S OWN ROW and saving to CSV
#columns = ['type', 'posts', 'comment']
#dftemp = pd.DataFrame(data=df, index=df.index, columns=columns)
##dftemp['comment'].fillna(0)
#dfcomments = pd.concat([pd.Series(row['comment'], row['posts'].split('|||')) for _, row in dftemp.iterrows()]).reset_index()
#del dfcomments[0]
#dfcomments.reset_index()
#dfcomments.rename(columns={'index': 'comment'})
#dfcomments.to_csv("komentarji.csv", sep=';')
########

## Začetna vizualizacija za št besed glede na komentar določenega MBTI tipa
plt.figure(figsize=(10,8))
sns.swarmplot("type", "words_per_person", data=df)

#x = df.groupby['words_per_person'].sort_values(ascending=False)
#sns.barplot("words_per_person", "type", data=df)

# =============================================================================
## Če želimo navaden barplot in posortiran
plotdf = pd.DataFrame(data=df)
del plotdf['variance_of_word_counts']
del plotdf['number_comments_per_person']
plotdf.groupby(['type']).median().sort_values("words_per_person").plot.bar()

## Poglejmo število vseh klasificiranih komentarjev glede na tip
df.groupby('type').agg({'type':'count'}).sort_values('type', ascending=False)
## ZANIMIVOST!!!! Glede na splošno populacijo, vidimo da je v našem 
## datasetu ravno obratna reprezentacija MBTI tipov.


## Kot lahko vidimo so tipi ESFJ, ESFP, ESTJ in ESTP zelo redki

## Dajmo ustvarit nov dataframe brez teh tipov 
#df2 = df[~df['type'].isin(['ESFJ','ESFP','ESTJ','ESTP'])]
df2 = df
#df2.head()

## Dajmo prešteti vse komentarje kateri vsebujejo linke na druge strani in vse komentarje ki sprašujejo
df2['http_per_comment'] = df2['posts'].apply(lambda x: x.count('http'))
## Vprašaji 
df2['qm_per_comment'] = df2['posts'].apply(lambda x: x.count('?'))

df2[['http_per_comment','qm_per_comment']]

# testdata.map(lambda x: x if (x < 30 or x > 60) else 0)
df2['qm_per_comment_additional'] = df2['posts'].apply(lambda x: x.count('?|||') + x.count('? ') + x.count(' ?'))
#df2.head(20)

## =============================================================================
## Iz pregleda lahko opazimo, da prvi komentar (index 0) vsebuje veliko linkov (24 "http" stringov).
## Iz tega lahko tudi sklepamo, da je qm_per_comment (na indexu 0) visok tudi zaradi tega, ker linki pravtako
## vsebujejo '?' znake.

## Poglejmo povprečje linkov glede na tip
print(df2.groupby('type').agg({'http_per_comment': 'mean'}).sort_values('http_per_comment', ascending=False))
## Poglejmo povprečje '?' znakov glede na tip
print(df2.groupby('type').agg({'qm_per_comment_additional': 'mean'}).sort_values('qm_per_comment_additional', ascending=False))
df2.head()

## Vizualizacija
plt.figure(figsize=(10,8))
sns.swarmplot("type", "qm_per_comment_additional", data=df2)

plt.figure(figsize=(10,8))
sns.swarmplot("type", "http_per_comment", data=df2)

# Not working
# df2['counted_words_per_person'] = df.groupby('words_per_person').agg({'words_per_person': 'sum'}).sort_values('words_per_person', ascending=False)


# =============================================================================
## TODO
## DEJMO ZDEJ POGLEDAT PO ***DOLŽINI*** KOMENTARJEV!
## DEJMO ZDRUŽIT ESFJ, ESFP, ESTJ in ESTP v **ESXX**
## ZA GRADNJO MODELA SLEDI ŠE VEKTORIZACIJA PODATKOV

## - Najdimo vektorje ki bodo pomagali razbiti. 4 utežene klasifikacije (Introvert - Extrovert, etc. etc. etc. Binarna klasifikacija)
## - Najti vektor, ki uspe binarno klasificirati.
## - Dve zadevi ki so problematični - dataset ni prezentativen kot v realni populaciji
## - Unconditional probability, transitional probability.

## Z nltk paketom
## Vsak izmed nas mora najdit 15-20 nekih featurjev, ki bi mogoče lahko opisali te binarne klasifikacije
## 4 axis plot -- Seštet vse words_per_person glede na I in E, etc etc etc.
## Mutidimensional classifier (več featurjev)
## Sentiment, dolžina komentarja, dolžina stavka.
# =============================================================================




#feature = 'qm_per_comment'
#dffunc = df.copy()
#dffunc = pd.concat([dffunc, pd.DataFrame({'ax{}'.format(i): df['type'].astype(str).str[i] for i in range(4)})], axis=1)
#
#def _grpstats(g):
#    tmp = pd.DataFrame({'mean': g.apply(np.nanmean, axis=0),
#                        'std': g.apply(np.std, axis=0)})
#    return tmp
#axioms = {i: _grpstats(dffunc[feature].groupby(dffunc['ax{}'.format(i)])) for i in range(4)}


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
#  example: rapidplot(df2, 'qm_per_comment_additional')
#
# =============================================================================
def rapidplot(df, feature):
    dffunc = df
    dffunc['1ax'] = df['type'].astype(str).str[0]
    dffunc['2ax'] = df['type'].astype(str).str[1]
    dffunc['3ax'] = df['type'].astype(str).str[2]
    dffunc['4ax'] = df['type'].astype(str).str[3]
    axiom1 = dffunc[[feature,'1ax']].groupby('1ax').apply(np.nanmean, axis=0).drop('1ax', axis=1, inplace=False).reset_index(drop=False)
    axiom2 = dffunc[[feature,'2ax']].groupby('2ax').apply(np.nanmean, axis=0).drop('2ax', axis=1, inplace=False).reset_index(drop=False)
    axiom3 = dffunc[[feature,'3ax']].groupby('3ax').apply(np.nanmean, axis=0).drop('3ax', axis=1, inplace=False).reset_index(drop=False)
    axiom4 = dffunc[[feature,'4ax']].groupby('4ax').apply(np.nanmean, axis=0).drop('4ax', axis=1, inplace=False).reset_index(drop=False)
    fig, ax = plt.subplots(2,2,figsize=(10,8))
    fig.suptitle('Kvaliteta vektorja = '+feature)
    sns.set(style="white", color_codes=True)
    sns.barplot(x=axiom1['1ax'], y=axiom1[feature], ax=ax[0,0])
    sns.barplot(x=axiom2['2ax'], y=axiom2[feature], ax=ax[0,1])
    sns.barplot(x=axiom3['3ax'], y=axiom3[feature], ax=ax[1,0])
    sns.barplot(x=axiom4['4ax'], y=axiom4[feature], ax=ax[1,1])
    
#rapidplot(df2, 'qm_per_comment')
    
def fq_barplot(df, feature):
    dffunc = pd.concat([df, pd.DataFrame({'ax{}'.format(i): df['type'].astype(str).str[i] for i in range(4)})], axis=1)
    fig, ax = plt.subplots(2,2,figsize=(10,8))
    ax = ax.flatten()
    fig.suptitle('Kvaliteta vektorja = '+feature)
    sns.set(style="white", color_codes=True)
    for i in range(4):
        sns.barplot(x='ax{}'.format(i), y=feature, data=dffunc, ax=ax[i], ci='sd')

#fq_barplot(df2, 'qm_per_comment')


def fq_swarmplot(df, feature):
    dffunc = pd.concat([df, pd.DataFrame({'ax{}'.format(i): df['type'].astype(str).str[i] for i in range(4)})], axis=1)
    fig, ax = plt.subplots(2,2,figsize=(10,8))
    ax = ax.flatten()
    fig.suptitle('Kvaliteta vektorja = '+feature)
    sns.set(style="white", color_codes=True)
    for i in range(4):
        sns.swarmplot(x='ax{}'.format(i), y=feature, data=dffunc, ax=ax[i])

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
    psts = pd.DataFrame({'post':pd.Series(psts)})
    psts['post_length'] = psts['post'].apply(lambda x: len(x))
    psts['uid'] = uid
    posts_df.append(psts)

posts_df = pd.concat(posts_df).reset_index(drop=False)
posts_df.to_csv('mbti_consolidated2.csv',index=False)


# =============================================================================

# =============================================================================
#                       NLTK knjižnica - delo
# =============================================================================


# =============================================================================
#                       ANŽE PEHARC FEATURES
# =============================================================================
#število ! na komentar
df_melt['!_per_comment_additional'] = df_melt['post'].apply(lambda x: x.count('!'))
#število I v komentarjih. Vidimo kolikokrat oseba omenja samo sebe.
df_melt['I_per_comment'] = df_melt['post'].apply(lambda x: x.count(' I ') + x.count("I'"))
#df_melt['I_per_comment'].mean()
#df_melt.head(10)
#kolikorat osebe zanikajo nekaj
df_melt['denied_words_per_comment'] = df_melt['post'].apply(lambda x: x.count("'t"))
#express love and compasion
df_melt['love_per_comment'] = df_melt['post'].apply(lambda x: x.count('love') + x.count('care') + x.count('feel'))
#število omenjevanj drugih ljudi
df_melt['WE&THEY_words_per_comment'] = df_melt['post'].apply(lambda x: x.count(" we ") + x.count("We ") + x.count(" they ") + x.count("They "))
#število predlogov
df_melt['predlogi_per_comment'] = df_melt['post'].apply(lambda x: x.count(" in ") + x.count(" on ") + x.count(" at ") + x.count(" since ") + x.count(" for ") + x.count(" by "))
#vprašalnice
df_melt['asking_words_per_comment'] = df_melt['post'].apply(lambda x: x.count(" how ") + x.count("How ") + x.count(" where ") + x.count("Where ") + x.count("When ") + x.count(" when ") + x.count("Who ") + x.count(" who "))
#smiley faces
df_melt['smiley_faces_per_comment'] = df_melt['post'].apply(lambda x: x.count(":)") + x.count(":D"))
#vejice in pike
df_melt['vejice_per_comment'] = df_melt['post'].apply(lambda x: x.count(",") + x.count("."))  
#velike črke
df_melt['uppercase_per_comment'] = df_melt['post'].apply(lambda x: len(re.findall(r'[A-Z]', x)))
#številke
df_melt['numbers_per_comment'] = df_melt['post'].apply(lambda x: len(re.findall(r'[0-9]', x)))
# =============================================================================
#                       KEVIN CVETEŽAR FEATURES
# =============================================================================
def stop_words(text):
    count = 0
    tokens = word_tokenize(text)
    for token in tokens:
        if token in stopwords.words('english'):
            count+=1;
    print(count)
    return count;

def lexical_diversity(text): 
    return len(set(text)) / len(text)


df_melt["cutenje"] = df2['posts'].apply(lambda x: x.upper().count('feel'.upper())+x.upper().count('feeling'.upper())+x.upper().count('feels'.upper())+x.upper().count('felt'.upper()))
df_melt["mislenje"] = df2['posts'].apply(lambda x: x.upper().count('think'.upper())+x.upper().count('thought'.upper())+x.upper().count('thinking'.upper())+x.upper().count('thinks'.upper()))
df_melt["love"] = df2['posts'].apply(lambda x: x.upper().count('love'.upper())+x.upper().count('loving'.upper())+x.upper().count('loves'.upper()))
df_melt["hate"] = df2['posts'].apply(lambda x: x.upper().count('hate'.upper())+x.upper().count('hating'.upper())+x.upper().count('hates'.upper())+x.upper().count('hated'.upper()))
df_melt["apolagetic"] = df2['posts'].apply(lambda x: x.upper().count('sorry'.upper())+x.upper().count('sry'.upper())+x.upper().count('apologies'.upper())+x.upper().count('apology'.upper()))
df_melt["lexical_diversity"] = df2['posts'].apply(lambda x: lexical_diversity(x))
df_melt["group_mentality"] = df2['posts'].apply(lambda x: x.upper().count('we'.upper())+x.upper().count('us'.upper())+x.upper().count('together'.upper())+x.upper().count('our'.upper())+x.upper().count('together'.upper()))
df_melt["thankfulness"] = df2['posts'].apply(lambda x: x.upper().count('thank'.upper())+x.upper().count('thanks'.upper()))
#df_melt["stop_words"] = df2['posts'].apply(lambda x: stop_words(x)) <---dolgo izvajanje

print(df_melt.groupby('type').agg({'cutenje': 'mean'}).sort_values('cutenje', ascending=False))
print(df_melt.groupby('type').agg({'mislenje': 'mean'}).sort_values('mislenje', ascending=False))
print(df_melt.groupby('type').agg({'love': 'mean'}).sort_values('love', ascending=False))
print(df_melt.groupby('type').agg({'hate': 'mean'}).sort_values('hate', ascending=False))
print(df_melt.groupby('type').agg({'apolagetic': 'mean'}).sort_values('apolagetic', ascending=False))
print(df_melt.groupby('type').agg({'lexical_diversity': 'mean'}).sort_values('lexical_diversity', ascending=False))
print(df_melt.groupby('type').agg({'group_mentality': 'mean'}).sort_values('group_mentality', ascending=False))
print(df_melt.groupby('type').agg({'thankfulness': 'mean'}).sort_values('thankfulness', ascending=False))
#print(df_melt.groupby('type').agg({'stop_words': 'mean'}).sort_values('stop_words', ascending=False))
# =============================================================================


# =============================================================================
#               STEMMING and LEMMATIZING moved to -> stem_lemmat.py
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
dfcon = pd.read_csv("mbti_consolidated2.csv", encoding="latin1")


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

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

n_instances = 100
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
len(subj_docs), len(obj_docs)


train_subj_docs = subj_docs[:80]
test_subj_docs = subj_docs[80:100]
train_obj_docs = obj_docs[:80]
test_obj_docs = obj_docs[80:100]
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
    
    
from nltk import tokenize
sentences = []
tuples = [tuple(x) for x in dflem.values]

lines_list = tokenize.sent_tokenize(tuples)

sentences.extend(lines_list)
sid = SentimentIntensityAnalyzer()
for sentence in sentences:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
         print('{0}: {1}, '.format(k, ss[k]), end='')







dfcon = dfcon.iloc[:10]
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer

setnan = SentimentAnalyzer()
sentinan = SentimentIntensityAnalyzer()

for i, row in dfcon.iterrows():
#    ss = setnan.evaluate(row['post'])
    ss = sentinan.polarity_scores(row['post'])
    print(row['post'], ss)









