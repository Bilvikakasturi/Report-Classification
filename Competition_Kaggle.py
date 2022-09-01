import time
import plotly.express as px
import pandas as pd
import re, string
import nltk
import pyproj
from nltk.corpus import stopwords
import numpy as np
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from xgboost import XGBClassifier

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
def plot_column_distribution(df, column):

    column_val_df = df[column].value_counts().to_frame().reset_index()
    column_val_df.columns = [column, 'count']
    fig = px.bar(data_frame=column_val_df, x=column, y='count')
    fig.update_layout(
        autosize=True,
        height=600,
        hovermode='closest',
        showlegend=True,
        margin=dict(l=10, r=10, t=30, b=0)
    )

    fig.show()
    return None

def xy_to_lonlat(x, y):
    project_latlon = pyproj.Proj(proj='latlong', datum='WGS84')
    projectxy = pyproj.Proj(proj="utm", zone=33, datum='WGS84')
    lon_lat = pyproj.transform(projectxy, project_latlon, x, y)
    return lon_lat[0].tolist(), lon_lat[1].tolist()
def extract_year(date):
    return int(date.split('/')[2])
def extract_date(time):
    return time.split(' ')[0]
def extract_month(date):
    return int(date.split('/')[1])
def extract_season(month):
    if month in [4, 5, 6]:
        return 'summer'
    elif month in [7, 8, 9]:
        return 'rainy'
    elif month in [10, 11, 12]:
        return 'winter'
    return 'spring'
def extract_day(date):
    return int(date.split('/')[0])
df_crime_train = pd.read_csv('train.csv')
df_crime_train["NARRATIVE"].fillna(df_crime_train["CRIMETYPE"], inplace=True)

df_crime_test = pd.read_csv('test.csv')
df_crime_train['key'] = 'train'
df_crime_test['CRIMETYPE'] = 'null'
df_crime_test['key'] = 'test'
df_crime_test["NARRATIVE"].fillna("BTFV", inplace = True)
df_crime_train = df_crime_train.append(df_crime_test, ignore_index=True)
Lat1, Lon1 = (xy_to_lonlat(df_crime_train['X'], df_crime_train['Y']))

df_crime_train['Lat'] = Lat1
df_crime_train['Lon'] = Lon1
df_crime_train['Date'] = df_crime_train['BEGDATE'].apply(extract_date)
df_crime_train['Year'] = df_crime_train['Date'].apply(extract_year)
df_crime_train['Month'] = df_crime_train['Date'].apply(extract_month)
df_crime_train['Day'] = df_crime_train['Date'].apply(extract_day)
df_crime_train['Season'] = df_crime_train['Month'].apply(extract_season)
print(df_crime_train[['Lat', 'Lon']].describe())
print(df_crime_train[['Lat', 'Lon']].describe())
plot_column_distribution(df=df_crime_train, column='NARRATIVE')
plot_column_distribution(df=df_crime_train, column='Year')
plot_column_distribution(df=df_crime_train, column='Season')
df_crime_train = df_crime_train.dropna(subset=['NARRATIVE'])
df_crime_train['word_count'] = df_crime_train['NARRATIVE'].apply(lambda x: len(str(x).split()))
print(df_crime_train[df_crime_train['CRIMETYPE'] == 'BTFV']['word_count'].mean())
print(df_crime_train[df_crime_train['CRIMETYPE'] == 'BURG']['word_count'].mean())

wordlemm = WordNetLemmatizer()
def preprocessstring(textstring):
    # print(text)
    textstring = textstring.lower()
    textstring = textstring.strip()
    textstring = re.compile('<.*?>').sub('', textstring)
    textstring = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', textstring)
    textstring = re.sub('\s+', ' ', textstring)
    textstring = re.sub(r'\[[0-9]*\]', ' ', textstring)
    textstring = re.sub(r'[^\w\s]', '', str(textstring).lower().strip())
    textstring = re.sub(r'\d', ' ', textstring)
    textstring = re.sub(r'\s+', ' ', textstring)
    return textstring

def get_wordnet_pos(tag):
    if tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('J'):
        return wordnet.ADJ
    else:
        return wordnet.NOUN
def stopword(string):
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

def lemmatizer(string):
    wrd_pos_tags = nltk.pos_tag(word_tokenize(string))
    a = [wordlemm.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in
         enumerate(wrd_pos_tags)]
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocessstring(string)))
df_crime_train['text_clean'] = df_crime_train['NARRATIVE'].apply(lambda x: finalpreprocess(x))
print(df_crime_train.head())

df_crime_test = df_crime_train[df_crime_train['key'] == 'test']
df_crime_train = df_crime_train[df_crime_train['key'] == 'train']

X_train = df_crime_train['NARRATIVE'].squeeze()
y_train = df_crime_train['CRIMETYPE'].squeeze()
X_test = df_crime_test['NARRATIVE'].squeeze()
y_test = df_crime_test['CRIMETYPE'].squeeze()
X_train_token = [nltk.word_tokenize(i) for i in X_train]
X_test_token = [nltk.word_tokenize(i) for i in X_test]

tfi_df_vectorizer = TfidfVectorizer(use_idf=True)
X_train_vector_tfidf_transform = tfi_df_vectorizer.fit_transform(X_train)
X_test_vector_tfidf_transform = tfi_df_vectorizer.transform(X_test)

class MnEmbeddingVector(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(word2vec.values())))

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X])

    def fit(self, X, y):
        return self


df_crime_train['text_clean_to_vector'] = [nltk.word_tokenize(i) for i in df_crime_train['text_clean']]
model = Word2Vec(df_crime_train['text_clean_to_vector'], min_count=1)
word2vector = dict(zip(model.wv.index_to_key, model.wv.vectors))
modelword = MnEmbeddingVector(word2vector)
X_train_vectors_word_vect = modelword.transform(X_train_token)
X_test_vectors_w2v = modelword.transform(X_test_token)
df_test_drop = df_crime_test.dropna(subset=['NARRATIVE'])
print('Till vectorization sucessfull')

logireg_tfidf = LogisticRegression(solver='liblinear', C=10, penalty='l2')
logireg_tfidf.fit(X_train_vector_tfidf_transform, y_train)
y_predict_lg_tf = logireg_tfidf.predict(X_test_vector_tfidf_transform)
df_test_drop['id'] = pd.to_numeric(df_test_drop['id'])
my_submission_LR = pd.DataFrame({'id': df_test_drop.id, 'CRIMETYPE': y_predict_lg_tf})
my_submission_LR.to_csv('LogisticRegression.csv', index=False)
print('LogisticRegression done-')

logreg_wrd2vec = LogisticRegression(solver='liblinear', C=10, penalty='l2')
logreg_wrd2vec.fit(X_train_vectors_word_vect, y_train)  # model
y_predict_w2v = logreg_wrd2vec.predict(X_test_vectors_w2v)
df_test_drop['id'] = pd.to_numeric(df_test_drop['id'])
my_submission_WV = pd.DataFrame({'id': df_test_drop.id, 'CRIMETYPE': y_predict_w2v})
my_submission_WV.to_csv('LogisticRegression_w2v.csv', index=False)
print('LogisticRegression word to vec done')

naivb_tfidf = MultinomialNB()
naivb_tfidf.fit(X_train_vector_tfidf_transform, y_train)
y_predict_nv = naivb_tfidf.predict(X_test_vector_tfidf_transform)
df_test_drop['id'] = pd.to_numeric(df_test_drop['id'])
my_submission_NV = pd.DataFrame({'id': df_test_drop.id, 'CRIMETYPE': y_predict_nv})
my_submission_NV.to_csv('NV.csv', index=False)
print('NV done')

rf_model = RandomForestClassifier()
rf_model = rf_model.fit(X_train_vector_tfidf_transform, y_train)
y_pred_rf = rf_model.predict(X_test_vector_tfidf_transform)
df_test_drop['id'] = pd.to_numeric(df_test_drop['id'])
my_submission = pd.DataFrame({'id': df_test_drop.id, 'CRIMETYPE': y_pred_rf})
my_submission.to_csv('rf.csv', index=False)

xgb = XGBClassifier(n_estimators=100)
xgb=xgb.fit(X_train_vector_tfidf_transform, y_train)
preds = xgb.predict(X_test_vector_tfidf_transform)
df_test_drop['id'] = pd.to_numeric(df_test_drop['id'])
my_submission_xg = pd.DataFrame({'id': df_test_drop.id, 'CRIMETYPE': preds})
my_submission_xg.to_csv('xg.csv', index=False)



from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

estimator_list = [
    ('Random Forest', rf_model),
    ('Naive Bayes', naivb_tfidf),
    ('XGBoost',xgb),
]
stack_model = StackingClassifier(
    estimators=estimator_list, final_estimator=LogisticRegression()
)
stack_model.fit(X_train_vector_tfidf_transform, y_train.values.ravel())
y_predict_final = stack_model.predict(X_test_vector_tfidf_transform)
pd.DataFrame(y_predict_final).to_csv("Final.csv")



