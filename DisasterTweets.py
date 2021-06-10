import pandas as pd
import numpy as np
import string
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os

# setting working directory
# os.chdir()
# os.getcwd()

pd.set_option('display.max_colwidth', None)

#####################################################################################################################

# # READING DATA FROM CSV

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


#####################################################################################################################

# # MISLABELED DUPLICATES
# Source: https://www.kaggle.com/dmitri9149/transformer-simple-baseline-model-s

df_mislabeled = train_data.groupby(['text']).nunique().sort_values(by='target', ascending=False)
df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']
index_misl = df_mislabeled.index.tolist()
lenght = len(index_misl)
train_nu_target = train_data[train_data['text'].isin(index_misl)].sort_values(by='text')
num_records = train_nu_target.shape[0]
print(f"There are {lenght} equivalence classes with mislabelling")
# each of them respresent a class with min 2 elements
print(
    f"There are {num_records} records in train set which generate {lenght} equivalence classes with mislabelling (raw "
    f"text, no cleaning)")

copy = train_nu_target.copy()
classes = copy.groupby('text').agg({'keyword': np.size, 'target': np.mean}).rename(
    columns={'keyword': 'Number of records in train set', 'target': 'Target mean'})

classes.sort_values('Number of records in train set', ascending=False).head(20)

# Majority voting: If Target mean is lower or equal 0.5 , relabel it to 0, otherwise to 1.
majority_df = train_nu_target.groupby(['text'])['target'].mean()


def relabel(r, majority_index):
    ind = ''
    if r['text'] in majority_index:
        ind = r['text']
        #        print(ind)
        if majority_df[ind] <= 0.5:
            return 0
        else:
            return 1
    else:
        return r['target']


train_data['target'] = train_data.apply(lambda row: relabel(row, majority_df.index), axis=1)
new_df = train_data[train_data['text'].isin(majority_df.index)].sort_values(['target', 'text'], ascending=[False, True])
new_df.head(15)
print(len(new_df))

# checking it again...
df_mislabeled = train_data.groupby(['text']).nunique().sort_values(by='target', ascending=False)
df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']
index_misl = df_mislabeled.index.tolist()
print(len(index_misl))


#####################################################################################################################

# REMOVING  SPECIAL CHARACTERS
# Got this list from here:
# https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert?scriptVersionId=28164619
def remove_special_characters(text, remove_digits=False):
    text = re.sub(r"\x89Û_", "", text)
    text = re.sub(r"\x89ÛÒ", "", text)
    text = re.sub(r"\x89ÛÓ", "", text)
    text = re.sub(r"\x89ÛÏWhen", "When", text)
    text = re.sub(r"\x89ÛÏ", "", text)
    text = re.sub(r"China\x89Ûªs", "China's", text)
    text = re.sub(r"let\x89Ûªs", "let's", text)
    text = re.sub(r"\x89Û÷", "", text)
    text = re.sub(r"\x89Ûª", "", text)
    text = re.sub(r"\x89Û\x9d", "", text)
    text = re.sub(r"å_", "", text)
    text = re.sub(r"\x89Û¢", "", text)
    text = re.sub(r"\x89Û¢åÊ", "", text)
    text = re.sub(r"fromåÊwounds", "from wounds", text)
    text = re.sub(r"åÊ", "", text)
    text = re.sub(r"åÈ", "", text)
    text = re.sub(r"JapÌ_n", "Japan", text)
    text = re.sub(r"Ì©", "e", text)
    text = re.sub(r"å¨", "", text)
    text = re.sub(r"SuruÌ¤", "Suruc", text)
    text = re.sub(r"åÇ", "", text)
    text = re.sub(r"å£3million", "3 million", text)
    text = re.sub(r"åÀ", "", text)
    text = re.sub(r"don\x89Ûªt", "do not", text)
    text = re.sub(r"I\x89Ûªm", "I am", text)
    text = re.sub(r"you\x89Ûªve", "you have", text)
    text = re.sub(r"it\x89Ûªs", "it is", text)
    text = re.sub(r"doesn\x89Ûªt", "does not", text)
    text = re.sub(r"It\x89Ûªs", "It is", text)
    text = re.sub(r"Here\x89Ûªs", "Here is", text)
    text = re.sub(r"I\x89Ûªve", "I have", text)
    text = re.sub(r"can\x89Ûªt", "cannot", text)
    text = re.sub(r"wouldn\x89Ûªt", "would not", text)
    text = re.sub(r"That\x89Ûªs", "That is", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"You\x89Ûªre", "You are", text)
    text = re.sub(r"Don\x89Ûªt", "Do not", text)
    text = re.sub(r"Can\x89Ûªt", "Cannot", text)
    text = re.sub(r"you\x89Ûªll", "you will", text)
    text = re.sub(r"I\x89Ûªd", "I would", text)
    text = re.sub(r"donå«t", "do not", text)
    # Character entity references
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&amp;", "&", text)
    return text


train_data['text'] = train_data['text'].apply(lambda x: remove_special_characters(x, True))
test_data['text'] = test_data['text'].apply(lambda x: remove_special_characters(x, True))


#####################################################################################################################

# REMOVING ACCENTED CHARACTERS

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


train_data['text'] = train_data['text'].apply(lambda x: remove_accented_chars(x))
test_data['text'] = test_data['text'].apply(lambda x: remove_accented_chars(x))

print(remove_accented_chars('Sómě Áccěntěd těxt'))


#####################################################################################################################

# PREPROCESSING TEXT


def clean_text(text):
    text = re.sub('\[.*?\]', '', text)  # remove text in square brackets
    text = re.sub('<.*?>+', '', text)  # remove html tags
    text = re.sub('\n', ' ', text)
    text = re.sub('[‘’“”…]', '', text)  # remove those fancy quotes
    text = text.lower()
    return text


train_data['text'] = train_data['text'].apply(lambda x: clean_text(x))
test_data['text'] = test_data['text'].apply(lambda x: clean_text(x))


#####################################################################################################################

# CONTRACTIONS


import contractions

train_data['text'] = train_data['text'].apply(lambda x: contractions.fix(x))
test_data['text'] = test_data['text'].apply(lambda x: contractions.fix(x))

print(contractions.fix("you've"))
print(contractions.fix("he's"))
print(train_data.text.head(10))


#####################################################################################################################

# ABBREVIATIONS, INTERNET SLANGS
# csv file is downloaded by R script with web scraping: noslang.com/dictionary/
# function source: https://www.kaggle.com/bkassem/disastor-tweets-cleaning


import csv

with open('abbreviations_dict_v5.csv', mode='r') as infile:
    reader = csv.reader(infile)
    abbr_dict = {rows[0]: rows[1] for rows in reader}


def replace_abbr(txt, slang):
    ctxt = re.sub(r'\s+', ' ', txt)
    res = []
    for tok in ctxt.split():
        if tok.lower() in slang:
            res.append(slang[tok.lower()])
        else:
            res.append(tok)
    res = ' '.join(res)
    return res.strip()


train_data['text'] = train_data['text'].apply(lambda x: replace_abbr(x, abbr_dict))
test_data['text'] = test_data['text'].apply(lambda x: replace_abbr(x, abbr_dict))

print(train_data.text.head())


#####################################################################################################################

# TWEET PREPROCESSOR

# installing tweet-preprocessor
# https://github.com/s/preprocessor
# https://pypi.org/project/tweet-preprocessor/
import preprocessor as p

# Removes URLS, Emojis, Smileys, Mentions, Numbers and Reserved Words like RT & FAV
# I'm keeping the hashtags by setting the options for all the others;
p.set_options(p.OPT.RESERVED, p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.MENTION, p.OPT.NUMBER)
train_data['text'] = train_data['text'].apply(lambda x: p.clean(x))
test_data['text'] = test_data['text'].apply(lambda x: p.clean(x))

print(train_data.text.head())


#####################################################################################################################

# HASHTAG SEGMENTATION
# https://pypi.org/project/wordsegment/

from wordsegment import load, segment

load()


def word_segmentation(text):
    temp_text = text
    for x in re.finditer(r'#(\w+)', text):
        temp_text = temp_text.replace(x.group(), ' '.join(segment(x.group().replace("#", ""))))
    return temp_text


train_data['text'] = train_data['text'].apply(lambda x: word_segmentation(x))
test_data['text'] = test_data['text'].apply(lambda x: word_segmentation(x))

print(train_data.text.head())


#####################################################################################################################

# REPLACING SOME MISSPELLED WORDS/EXPANDING ACRONYMS


def replace_text(text):
    # some misspelled words
    text = re.sub(r"((?<!\w)|^)annonymous((?!\w)|$)", "anonymous", text)
    text = re.sub(r"((?<!\w)|^)ambulancewe((?!\w)|$)", "ambulance", text)
    text = re.sub(r"((?<!\w)|^)traumatised((?!\w)|$)", "traumatized", text)
    text = re.sub(r"((?<!\w)|^)laighign((?!\w)|$)", "laughing", text)
    text = re.sub(r"((?<!\w)|^)thursd((?!\w)|$)", "thursday", text)
    text = re.sub(r"((?<!\w)|^)recentlu((?!\w)|$)", "recently", text)
    text = re.sub(r"((?<!\w)|^)ph0tos((?!\w)|$)", "photos", text)
    text = re.sub(r"((?<!\w)|^)exp0sed((?!\w)|$)", "exposed", text)
    text = re.sub(r"((?<!\w)|^)mì¼sica((?!\w)|$)", "music", text)
    # acronyms
    text = re.sub(r"((?<!\w)|^)usagov((?!\w)|$)", "usa government", text)
    text = re.sub(r"((?<!\w)|^)bebacksoon((?!\w)|$)", "be back soon", text)
    text = re.sub(r"((?<!\w)|^)hawaiianpaddlesports((?!\w)|$)", "hawaiian paddle sports", text)
    text = re.sub(r"((?<!\w)|^)whitehouse((?!\w)|$)", "white house", text)
    text = re.sub(r"((?<!\w)|^)ptsd((?!\w)|$)", "post traumatic stress disorder", text)
    text = re.sub(r"((?<!\w)|^)nyt((?!\w)|$)", "new york times", text)
    text = re.sub(r"((?<!\w)|^)nytimes((?!\w)|$)", "new york times", text)
    text = re.sub(r"((?<!\w)|^)mh370((?!\w)|$)", "malaysia airlines flight", text)
    text = re.sub(r"((?<!\w)|^)okwx((?!\w)|$)", "oklahoma city weather", text)
    text = re.sub(r"((?<!\w)|^)arwx((?!\w)|$)", "arkansas weather", text)
    text = re.sub(r"((?<!\w)|^)gawx((?!\w)|$)", "georgia weather", text)
    text = re.sub(r"((?<!\w)|^)scwx((?!\w)|$)", "south carolina weather", text)
    text = re.sub(r"((?<!\w)|^)cawx((?!\w)|$)", "california weather", text)
    text = re.sub(r"((?<!\w)|^)tnwx((?!\w)|$)", "tennessee weather", text)
    text = re.sub(r"((?<!\w)|^)azwx((?!\w)|$)", "arizona weather", text)
    text = re.sub(r"((?<!\w)|^)alwx((?!\w)|$)", "alabama weather", text)
    text = re.sub(r"((?<!\w)|^)tafs((?!\w)|$)", "terminal aerodrome forecast", text)
    text = re.sub(r"((?<!\w)|^)wordpressdotcom((?!\w)|$)", "wordpress", text)
    text = re.sub(r"((?<!\w)|^)usnwsgov((?!\w)|$)", "united states national weather service", text)
    #
    text = re.sub(r'\s+', ' ', text)  # removing extra whitespaces
    return text.strip()


train_data['text'] = train_data['text'].apply(lambda x: replace_text(x))
test_data['text'] = test_data['text'].apply(lambda x: replace_text(x))

print(train_data.text.head())


#####################################################################################################################

# REMOVING NUMBERS AND PUNCTUATIONS


def remove_punctuation_number(text, remove_digits=False):
    text = re.sub(r'[%s]' % re.escape(''.join(string.punctuation)), r' ', text)  # remove punctuation
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


train_data['text'] = train_data['text'].apply(lambda x: remove_punctuation_number(x, True))
test_data['text'] = test_data['text'].apply(lambda x: remove_punctuation_number(x, True))

remove_punctuation_number("Well this was fun! What do you think? 123#@!", remove_digits=True)

#####################################################################################################################

# REMOVING STOPWORDS


nltk.download('wordnet')
nltk.download('stopwords')
stoplist = set(stopwords.words('english'))

train_data['text'] = train_data['text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stoplist)]))
test_data['text'] = test_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stoplist)]))

test_data.head()
train_data.head()


#####################################################################################################################

# TOKENIZER


nltk.download('punkt')

token = nltk.tokenize.RegexpTokenizer(r'\w+')

train_data['text'] = train_data['text'].apply(lambda x: token.tokenize(x))
test_data['text'] = test_data['text'].apply(lambda x: token.tokenize(x))

print(train_data.text.head())


#####################################################################################################################

# LEMMATIZER

def lemmatizer(text):
    lem_text = ' '.join(WordNetLemmatizer().lemmatize(i) for i in text)
    return lem_text


train_data['text'] = train_data['text'].apply(lambda x: lemmatizer(x))
test_data['text'] = test_data['text'].apply(lambda x: lemmatizer(x))

print(train_data.text.head())


#####################################################################################################################

# REMOVING SINGLE LETTER & 2 LETTER WORDS


def remove_single_characters(text):
    new_text = ' '.join([w for w in text.split() if len(w) > 2])
    return new_text


train_data['text'] = train_data['text'].apply(lambda x: remove_single_characters(x))
test_data['text'] = test_data['text'].apply(lambda x: remove_single_characters(x))


#####################################################################################################################

# FEATURE VECTORIZATION USING TFIDF

Y = list(train_data['target'])
X_train, X_test, y_train, y_test = train_test_split(train_data['text'], Y, test_size=0.0001, random_state=0)
print(len(X_train), " ", len(X_test))

tfidf_vectorizer = TfidfVectorizer(stop_words="english", decode_error="ignore")
tfidf_vectorizer.fit(X_train)

# SVM MODEL: SCORING .80416
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

cls = SVC()
cls.fit(tfidf_vectorizer.transform(X_train), y_train)
y_pred = cls.predict(tfidf_vectorizer.transform(X_test))

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

ans = cls.predict(tfidf_vectorizer.transform(test_data['text']))

sample_submission = pd.read_csv("sample_submission.csv")
sample_submission["target"] = ans
sample_submission.to_csv("submission_svm.csv", index=False)
sample_submission.head()
