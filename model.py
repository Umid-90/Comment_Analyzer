
import pandas as pd
from tqdm.auto import tqdm
import re
from lightgbm import LGBMClassifier
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
import nltk
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords


# In[2]:


df_reviews = pd.read_csv('comments.csv')
df_reviews.head(10)


# In[3]:


text = re.sub(r"[^a-zA-Z\']", " ", df_reviews['comment_text'][0])
text


# In[4]:


# target value for train data set
train_target = df_reviews['toxic']


# In[5]:

#We assume all models below accepts texts in lowercase
#and without any digits, punctuations marks etc.
def clear_text(text):

    text = re.sub(r"[^a-zA-Z\']", " ", text)

    return " ".join(text.split())

df_reviews['comment_text_norm'] = df_reviews['comment_text'].apply(lambda x : clear_text(x) )
df_reviews['comment_text_norm'].head(10)


# In[6]:


def text_preprocessing_3(text):

    doc = nlp(text)
    #tokens = [token.lemma_ for token in doc if not token.is_stop]
    tokens = [token.lemma_ for token in doc]

    return ' '.join(tokens)

# tokenizing and lemmatizing
df_reviews['comment_text_norm_prep'] = df_reviews['comment_text_norm'].apply(lambda x: text_preprocessing_3(x))

# converting the words into features using scapy
corpus_train = df_reviews['comment_text_norm_prep']


stop_words = set(nltk_stopwords.words('english'))
count_tf_idf = TfidfVectorizer(stop_words=stop_words)

train_features = count_tf_idf.fit_transform(corpus_train)

print("The TF-IDF matrix size for train:", train_features.shape)


# model
lgb = LGBMClassifier(random_state=121)
# fit the model
model = lgb.fit(train_features,train_target)

from sklearn.metrics import f1_score
predict = model.predict(train_features)
f1_score(train_target, predict, average=None)

# Saving model to disk
from joblib import dump

dump(lgb, 'Toks.joblib')
dump(count_tf_idf,'tf_idf.joblib')

# load model
from joblib import load
model = load('Toks.joblib')
vectorizer = load('tf_idf.joblib')
