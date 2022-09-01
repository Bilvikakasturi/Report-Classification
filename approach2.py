
from sklearn.neighbors import KNeighborsClassifier
from collections import OrderedDict
import re
import string

import nltk
import numpy as np
import pandas as pd

from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

trn_data = pd.read_csv("train.csv")
trn_data = trn_data.dropna(subset=['NARRATIVE'])

print(trn_data.head())

print(trn_data.shape)

target = trn_data["CRIMETYPE"].unique()
print(target)
tst_data = pd.read_csv("test.csv")
tst_data = tst_data.dropna(subset=['NARRATIVE'])
print(tst_data.head())
print(tst_data.shape)

d_dictionary = {}
count = 1
for data in target:
    d_dictionary[data] = count
    count += 1
trn_data["CRIMETYPE"] = trn_data["CRIMETYPE"].replace(d_dictionary)

def extract_date(time):
    """Extract data from time"""
    return time.split(' ')[0]


def extract_year(date):
    """Extract year from date"""
    return int(date.split('/')[2])


def extract_month(date):
    """Extract month from date"""
    return int(date.split('/')[1])


def extract_day(date):
    """Extract day from date"""
    return int(date.split('/')[0])


def extract_season(month):
    """Determine season from month"""
    if month in [4, 5, 6]:
        return 'summer'
    elif month in [7, 8, 9]:
        return 'rainy'
    elif month in [10, 11, 12]:
        return 'winter'
    return 'spring'


def preprocess(txt):

    txt = txt.lower()
    txt = txt.strip()
    txt = re.compile('<.*?>').sub('', txt)
    txt = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', txt)
    txt = re.sub('\s+', ' ', txt)
    txt = re.sub(r'\[[0-9]*\]', ' ', txt)
    txt = re.sub(r'[^\w\s]', '', str(txt).lower().strip())
    txt = re.sub(r'\d', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt


def stopword(string):
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)


def title_text(text):
    """Title the text"""
    if isinstance(text, str):
        text = text.title()
        return text
    return text


wl = WordNetLemmatizer()


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string))
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in
         enumerate(word_pos_tags)]
    return " ".join(a)


def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))


trn_data['clean_text'] = trn_data['NARRATIVE'].apply(lambda x: finalpreprocess(x))
trn_data['clean_text_tok'] = [nltk.word_tokenize(i) for i in trn_data['clean_text']]

trn_data['Date'] = trn_data['BEGDATE'].apply(extract_date)
trn_data['Year'] = trn_data['Date'].apply(extract_year)
trn_data['Month'] = trn_data['Date'].apply(extract_month)
trn_data['Day'] = trn_data['Date'].apply(extract_day)
trn_data['Season'] = trn_data['Month'].apply(extract_season)
tst_data['clean_text'] = tst_data['NARRATIVE'].apply(lambda x: finalpreprocess(x))
tst_data['clean_text_tok'] = [nltk.word_tokenize(i) for i in tst_data['clean_text']]

tst_data['Date'] = tst_data['BEGDATE'].apply(extract_date)
tst_data['Year'] = tst_data['Date'].apply(extract_year)
tst_data['Month'] = tst_data['Date'].apply(extract_month)
tst_data['Day'] = tst_data['Date'].apply(extract_day)
tst_data['Season'] = tst_data['Month'].apply(extract_season)
print(tst_data.head())
columns_trn = trn_data.columns
print(columns_trn)
columns_tst = tst_data.columns
print(columns_tst)

clmn = columns_trn.drop("id")
print(clmn)
train_data_new = trn_data[clmn]
print(train_data_new.head())
print(train_data_new.describe())
corr = train_data_new.corr()
print(corr["CRIMETYPE"])
skew = train_data_new.skew()
print(skew)
features = ["Day", "NARRATIVE",  "X", "Y",'Date','Year','Month','Day','Season','clean_text_tok']
X_train = trn_data[features]
y_train = trn_data["CRIMETYPE"]
X_test = tst_data[features]


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

data_dict_new = OrderedDict(sorted(d_dictionary.items()))

print(type(predictions))
result_datafrm = pd.DataFrame({
    "id": tst_data["id"]
})
for key, value in data_dict_new.items():
    result_datafrm[key] = 0
count = 0
for item in predictions:
    for key, value in d_dictionary.items():
        if value == item:
            result_datafrm[key][count] = 1
    count += 1
print(predictions)
print(result_datafrm)
result_datafrm.to_csv("submission_knn.csv", index=False)

