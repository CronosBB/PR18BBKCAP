# PR18BBKCAP
Project work for Data Mining course at University of Ljubljana - 2018

Slovene only:

## Avtorji
- Blaž Blažinčič
- Kevin Cvetežar
- Anže Peharc

# Izbor podatkovne množice
Za projektno nalogo pri Podatkovnem Rudarjenju FRI Ljubljana smo si izbrali "(MBTI) Myers-Briggs Personality Type Dataset" (https://goo.gl/hqd5Nu). "Myers Briggs Type Indicator" je sistem klasifikacije osebnostnega karakterja, kateri deli vsakogar na 16 osebnostnih tipov skozi 4 axiome:

    Introversion (I) – Extroversion (E)
    Intuition (N) – Sensing (S)
    Thinking (T) – Feeling (F)
    Judging (J) – Perceiving (P)

Torej v primeru nekoga, ki se nagiba k "Extroversion", "Intuition", "Thinking" ter "Perceiving" bi označili kot ENTP v MBTI testu osebnosti. Obstaja veliko elementov, ki temeljijo na osebnosti, ki bi lahko modelirali ali opisali želje ali vedenje te osebe na podlagi oznake.
Je eden izmed najbolj priljubljenih testov osebnosti na svetu. Uporablja se v podjetjih, na spletu, za zabavo, za raziskave in še veliko več. Na spletu lahko najdemo veliko različnih uporab tega testa in lahko rečemo, da je ta test osebnosti še vedno zelo veljaven in uporaben.

## Oblika podatkov
Kaggle zbirka podatkov, ki jih bomo uporabili vsebuje 8676 vrstic podatkov. Vsaka vrstica vsebuje:

    Tip (MBTI koda z 4 črkami, ki opisuje osebnost) (STRING)
    Del vsakega od zadnjih 50 komentarjev, ki so jih objavili (vsak vnos je ločen z "|||") (STRING)

## Naša vprašanja oz. primeri uporabe zbirke podatkov

- Ali lahko z ML-jem ocenimo veljavnost MBTI testa osebnosti in njegove sposobnosti pri napovedovanju osebnosti na spletu?
- Ali lahko ustvarimo ML model, ki bo lahko najbolj uspešno napovedal MBTI osebnost osebe glede na njegove/njene zapise oz. komentarje?
    
Žal vsebuje zbirka podatkov komentarje le v angleščini, zatorej bo klasifikacija slovenskih komentarjev v osnovni fazi nemogoča. Kljub temu pa nam to predstavlja priložnost, da preko "web-scrapping"-a in po uspešno ustvarjenem in robustnim modelom poskušamo klasificirati uporabnike na svetovnem spletu.

Za evaluacijo naših MBTI osebnosti bomo uporabili še test osebnosti na spletni strani 16 Personalities (https://www.16personalities.com/free-personality-test), katera pravtako uporablja MBTI sistem klasifikacije osebnostnega karakterja in je ena izmed najbolj popularnih spletnih strani za napovedovanje MBTI karakterja.

# Inicializacija podatkov

Začnimo najprej z inicializacijo podatkov in branjem iz .csv datoteke. Za lažje delo in manipulacijo podatkov si bomo pomagali z knjižnicami **pandas** in **numpy** za delo z podatki ter **seaborn** in **matplotlib** za vizualizacijo podatkov. V prihodnje bomo za gradnjo modelov in evaluacijo klasifikacijske točnosti ter ostalo validacijo uporabili še popularno knjižnico **scikit-learn**.

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv('data/mbti_1.csv')
    
Najprej poglejmo strukturo naših podatkov v osnovnem podatkovnem nizu (*angl.* ***Dataframe***).

    >df.head(20)
        type                                              posts
    0   INFJ  'http://www.youtube.com/watch?v=qsXHcwe3krw|||...
    1   ENTP  'I'm finding the lack of me in these posts ver...
    2   INTP  'Good one  _____   https://www.youtube.com/wat...
    3   INTJ  'Dear INTP,   I enjoyed our conversation the o...
    4   ENTJ  'You're fired.|||That's another silly misconce...
    5   INTJ  '18/37 @.@|||Science  is not perfect. No scien...
    6   INFJ  'No, I can't draw on my own nails (haha). Thos...
    7   INTJ  'I tend to build up a collection of things on ...
    8   INFJ  I'm not sure, that's a good question. The dist...
    9   INTP  'https://www.youtube.com/watch?v=w8-egj0y8Qs||...
    10  INFJ  'One time my parents were fighting over my dad...
    11  ENFJ  'https://www.youtube.com/watch?v=PLAaiKvHvZs||...
    12  INFJ  'Joe santagato - ENTP|||ENFJ or  ENTP?   I'm n...
    13  INTJ  'Fair enough, if that's how you want to look a...
    14  INTP  'Basically this...  https://youtu.be/1pH5c1Jkh...
    15  INTP  'Your comment screams INTJ, bro. Especially th...
    16  INFJ  'some of these both excite and calm me:  BUTTS...
    17  INFP  'I think we do agree. I personally don't consi...
    18  INFJ  'I fully believe in the power of being a prote...
    19  INFP  'That's normal, it happens also to me. If I am...
    
Kot lahko razberemo iz začetnega podatkovnega niza vidimo, da vsebuje naša podatkovna množica primarno dva stolpca podatkov skupaj z indeks vrednostjo. To sta *type* in *posts*. Vsaka vrstica v tabeli pripada eni osebi. Ta oseba je klasificirana z določenim MBTI tipom osebnosti, komentarji pa se delijo glede na '|||' niz karakterjev.
    
    >df.describe
    [8675 rows x 2 columns]>
    
Kot lahko zgoraj vidimo imamo v podatkovni množici 8675 različnih "oseb".

## Pregled osnovnih podatkov in relevantnosti

V začetni observaciji podatkov nas mogoče zanima število oseb ter komentarjev glede na osebo. Kot lahko opazimo spodaj, se števila besed glede na osebo razlikujejo, število komentarjev je pa v večji meri enako. Razlog za to je sama struktura podatkovne množice in sicer maksimalno 50 komentarjev glede na osebo.

    >df['words_per_person'] = df['posts'].apply(lambda x: len(x.split()))
    >df['comments_per_person'] = df['posts'].apply(lambda x: len(x.split('|||')))
    
    >df[['words_per_person','comments_per_person']]
    
              words_per_person  comments_per_person
    0                  556                   50
    1                 1170                   50
    2                  836                   50
    3                 1064                   50
    4                  967                   50
    5                 1491                   50
    6                 1329                   50
    7                 1223                   50
    8                  738                   50
    9                 1233                   50
    10                1454                   50
    
V osnovi nas zanima distribucija podatkov glede na število besed na osebo za vsak MBTI tip posebej.

    plt.figure(figsize=(10,8))
    sns.swarmplot("type", "words_per_person", data=df)
    
![distribucija podatkov](figures/Figure_1%20-%20Words%20per%20person%20-%20type_fixed.png)

Iz osnovne distribucije glede na število besed/MBTI tip lahko razberemo nekaj zanimivih predpostavk.

Glede na distribucijo, so Introverzni tipi osebnosti (torej tisti kateri se začnejo z Ixxx) najbolj zgovorni v internetnem forumu, vsaj glede na našo podatkovno množico. Seveda je to lahko zaradi več razlogov, recimo tip foruma in njegovih uporabnikov. Trditi v tem koraku še ne moremo ničesar. Zanimivo je tudi, da so tipi ESTP, ESFP, ESTJ in ESFJ najmanj zastopani v naši podatkovni množici.

Kaj je še bolj zanimivo je tudi to, da je naša podatkovna množica skoraj popolnoma nasprotno sorazmerna glede na splošno populacijo, kot lahko vidimo spodaj.

![mbti_genpop](figures/MBTI%20types%20in%20general%20population.PNG)


## Zanimivosti komentarjev

Dajmo prešteti vse komentarje kateri vsebujejo linke na druge strani in vse komentarje, ki so vprašanja in poglejmo njihova povprečne vrednosti glede na tip osebnosti.

        df2 = df
        df2['http_per_comment'] = df2['posts'].apply(lambda x: x.count('http'))
        df2['qm_per_comment_additional'] = df2['posts'].apply(lambda x: x.count('?|||') + x.count('? ') + x.count(' ?'))

Izpišimo povprečno število vprašanj glede na tip osebnosti in kasneje spodaj distribucijo naše podatkovne množice.

       >print(df2.groupby('type').agg({'qm_per_comment_additional': 'mean'}).sort_values('qm_per_comment_additional', ascending=False))
       

              qm_per_comment_additional
        type                           
        ENTJ                   9.835498
        ESFP                   9.500000
        ESTP                   9.460674
        ENTP                   9.097810
        ENFP                   8.549630
        INTJ                   8.388634
        ENFJ                   8.310526
        INTP                   8.259969
        ESTJ                   8.128205
        ISTJ                   8.097561
        ISTP                   8.032641
        INFJ                   7.617007
        ESFJ                   7.476190
        ISFJ                   7.337349
        ISFP                   7.158672
        INFP                   6.906659


![qm_per_person](figures/Figure_2%20-%20QM%20per%20person%20-%20type.png)

Enako nas zanima povprečje spletnih povezav glede na tip osebnosti in kasneje spodaj distribucijo naše podatkovne množice.

    >print(df2.groupby('type').agg({'http_per_comment': 'mean'}).sort_values('http_per_comment', ascending=False))
    
              http_per_comment
    type                  
    ISFP          4.416974
    ISTP          4.050445
    INFP          3.771288
    INTP          3.538344
    ISFJ          3.530120
    INFJ          3.293878
    ESTP          3.235955
    INTJ          3.179652
    ISTJ          3.058537
    ESFP          2.770833
    ENFJ          2.663158
    ENTJ          2.658009
    ESTJ          2.641026
    ENFP          2.522963
    ENTP          2.413139
    ESFJ          1.357143


![http_per_person](figures/Figure_3%20-%20Http%20per%20person%20-%20type.png)

    
## Glavne začetne ugotovitve

Vsekakor zanimivo dejstvo je to, da imamo v naši podatkovni množici bistveno večjo reprezentacijo redkejših MBTI tipov kot pa kaže generalna porazdelitev glede na splošno populacijo. To nam predstavlja že nova in zanimiva vprašanja katera bomo skušali odgovoriti v nadaljevanju.



