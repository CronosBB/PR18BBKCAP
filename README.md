# PR18BBKCAP
Project work for Data Mining course at University of Ljubljana - 2018

Slovene only:

# Izbor podatkovne množice
Za projektno nalogo pri Podatkovnem Rudarjenju FRI Ljubljana smo si izbrali "(MBTI) Myers-Briggs Personality Type Dataset" (https://goo.gl/hqd5Nu). "Myers Briggs Type Indicator" je sistem klasifikacije osebnostnega karakterja, kateri deli vsakogar na 16 osebnostnih tipov skozi 4 axiome:

    Introversion (I) – Extroversion (E)
    Intuition (N) – Sensing (S)
    Thinking (T) – Feeling (F)
    Judging (J) – Perceiving (P)

Torej v primeru nekoga, ki se nagiba k "Extroversion", "Intuition", "Thinking" ter "Perceiving" bi označili kot ENTP v MBTI testu osebnosti. Obstaja veliko elementov, ki temeljijo na osebnosti, ki bi lahko modelirali ali opisali želje ali vedenje te osebe na podlagi oznake.
Je eden izmed najbolj priljubljenih testov osebnosti na svetu. Uporablja se v podjetjih, na spletu, za zabavo, za raziskave in še veliko več. Na spletu lahko najdemo veliko različnih uporab tega testa in lahko rečemo, da je ta test osebnosti še vedno zelo veljaven in uporaben.

# Oblika podatkov
Kaggle zbirka podatkov, ki jih bomo uporabili vsebuje 8676 vrstic podatkov. Vsaka vrstica vsebuje:

    Tip (MBTI koda z 4 črkami, ki opisuje osebnost) (STRING)
    Del vsakega od zadnjih 50 komentarjev, ki so jih objavili (vsak vnos je ločen z "|||") (STRING)

# Naša vprašanja oz. primeri uporabe zbirke podatkov

- Ali lahko z ML-jem ocenimo veljavnost MBTI testa osebnosti in njegove sposobnosti pri napovedovanju osebnosti na spletu?
- Ali lahko ustvarimo ML model, ki bo lahko najbolj uspešno napovedal MBTI osebnost osebe glede na njegove/njene zapise oz. komentarje?
    
Žal vsebuje zbirka podatkov komentarje le v angleščini, zatorej bo klasifikacija slovenskih komentarjev v osnovni fazi nemogoča. Kljub temu pa nam to predstavlja priložnost, da preko "web-scrapping"-a in po uspešno ustvarjenem in robustnim modelom poskušamo klasificirati uporabnike na svetovnem spletu.

Za evaluacijo naših MBTI osebnosti bomo uporabili še test osebnosti na spletni strani 16 Personalities (https://www.16personalities.com/free-personality-test), katera pravtako uporablja MBTI sistem klasifikacije osebnostnega karakterja in je ena izmed najbolj popularnih spletnih strani za napovedovanje MBTI karakterja.

## Inicializacija podatkov

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv('data/mbti_1.csv')
    df.head()


