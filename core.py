# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:39:56 2018

@author: CronosBB
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/mbti_1.csv')
df.head()

## Splitamo vsak komentar
def var_row(row):
    l = []
    for i in row.split('|||'):
        l.append(len(i.split()))
    return np.var(l)

## 
df['words_per_person'] = df['posts'].apply(lambda x: len(x.split()))
df['comments_per_person'] = df['posts'].apply(lambda x: len(x.split('|||')))

df[['words_per_person','comments_per_person']]

##
#df['words_per_comment_splitted'] = df['posts'].apply(lambda x: x.split())
df['variance_of_word_counts'] = df['posts'].apply(lambda x: var_row(x))
df.head(20)

df.describe

## Ustvari Series komentarjev
dfstr = pd.DataFrame(data=df['posts'])
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

x = df.groupby['words_per_person'].sort_values(ascending=False)
sns.barplot("words_per_person", "type", data=df)


## Če želimo navaden barplot in posortiran
plotdf = pd.DataFrame(data=df)
del plotdf['variance_of_word_counts']
del plotdf['comments_per_person']
plotdf.groupby(['type']).median().sort_values("words_per_person").plot.bar()

## Poglejmo število vseh klasificiranih komentarjev glede na tip
df.groupby('type').agg({'type':'count'}).sort_values('type', ascending=False)
## ZANIMIVOST!!!! Glede na splošno populacijo, vidimo da je v našem 
## datasetu ravno obratna reprezentacija MBTI tipov.


## Kot lahko vidimo so tipi ESFJ, ESFP, ESTJ in ESTP zelo redki

## Dajmo ustvarit nov dataframe brez teh tipov 
df2 = df[~df['type'].isin(['ESFJ','ESFP','ESTJ','ESTP'])]
df2.head()

## Dajmo prešteti vse komentarje kateri vsebujejo linke na druge strani in vse komentarje ki sprašujejo
df2['http_per_comment'] = df2['posts'].apply(lambda x: x.count('http'))

## Vprašaji 
df2['qm_per_comment'] = df2['posts'].apply(lambda x: x.count('?'))
# testdata.map(lambda x: x if (x < 30 or x > 60) else 0)
df2['qm_per_comment_additional'] = df2['posts'].apply(lambda x: x.count('?|||') + x.count('? ') + x.count(' ?'))
df2.head(20)


## Iz pregleda lahko opazimo, da prvi komentar (index 0) vsebuje veliko linkov (24 "http" stringov).
## Iz tega lahko tudi sklepamo, da je qm_per_comment (na indexu 0) visok tudi zaradi tega, ker linki pravtako
## vsebujejo '?' znake.

## Poglejmo povprečje linkov glede na tip
print(df2.groupby('type').agg({'http_per_comment': 'mean'}).sort_values('http_per_comment', ascending=False))
## Poglejmo povprečje '?' znakov glede na tip
print(df2.groupby('type').agg({'qm_per_comment': 'mean'}).sort_values('qm_per_comment', ascending=False))
df2.head()

## Vizualizacija
plt.figure(figsize=(10,8))
sns.swarmplot("type", "qm_per_comment_additional", data=df2)

sns.swarmplot("type", "http_per_comment", data=df2)

df['counted_words_per_comment'] = df.groupby('words_per_comment').agg({'words_per_comment': 'sum'}).sort_values('words_per_comment', ascending=False)

## TODO
## DEJMO ZDEJ POGLEDAT PO ***DOLŽINI*** KOMENTARJEV!
## DEJMO ZDRUŽIT ESFJ, ESFP, ESTJ in ESTP v **ESXX**
## ZA GRADNJO MODELA SLEDI ŠE VEKTORIZACIJA PODATKOV


























