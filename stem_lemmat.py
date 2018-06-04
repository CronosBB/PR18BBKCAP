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
import time
import re



stemmer = PorterStemmer() 
lemmatizer = WordNetLemmatizer() 
# =============================================================================
#               STEMMING the consolidated comments! - Time(minutes) : 47min
# =============================================================================
# Da deluje spodnji for i in range, je potrebno na novo prebrat dataset (konsolidiran!) - ker smo izgubili nekatere indexe zgoraj (z filtriranjem)
# Encoding je latin1, ker drugače dobimo UnicodeDecodeError
readdata = pd.read_csv("mbti_consolidated.csv", encoding="latin1")
corpus=[]


## NE POGANJAJ - Glej file mbti_corpus_stemmed.csv!!!
start=time.time()
for i in range(readdata.shape[0]):
    posts = re.sub('[^a-zA-z]',' ', str(readdata['post'][i]))
    posts = posts.lower()
    posts = posts.split() 
    posts = [stemmer.stem(word) for word in posts if not word in set(stopwords.words('english'))]
    posts = ' '.join(posts)
    corpus.append(posts)

elapsed=time.time()-start
minutes = elapsed / 60

print(minutes)

#Export to CSV
import csv
csvfile = "mbti_corpus_stemmed.csv"

#Assuming res is a flat list
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in corpus:
        writer.writerow([val])  

# =============================================================================

# =============================================================================
#               LEMMATIZING the consolidated comments! Time(Minutes): 49min
# =============================================================================
# Da deluje spodnji for i in range, je potrebno na novo prebrat dataset (konsolidiran!) - ker smo izgubili nekatere indexe zgoraj (z filtriranjem)
# Encoding je latin1, ker drugače dobimo UnicodeDecodeError
readdata = pd.read_csv("mbti_consolidated.csv", encoding="latin1")
corpus_lemmatized=[]


## NE POGANJAJ - Glej file mbti_corpus_lemmatized.csv!!!
start=time.time()
for i in range(readdata.shape[0]):
    posts = re.sub('[^a-zA-z]',' ', str(readdata['post'][i]))
    posts = posts.lower()
    posts = posts.split() 
    posts = [lemmatizer.lemmatize(word) for word in posts if not word in set(stopwords.words('english'))]
    posts = ' '.join(posts)
    corpus_lemmatized.append(posts)

elapsed=time.time()-start
minutes = elapsed / 60

print(minutes)

#Export to CSV
import csv
csvfile = "mbti_corpus_lemmatized.csv"

#Assuming res is a flat list
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in corpus_lemmatized:
        writer.writerow([val])  

# =============================================================================