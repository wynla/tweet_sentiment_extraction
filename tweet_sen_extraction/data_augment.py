import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import nltk
import re

from nltk import word_tokenize
from nltk.corpus import stopwords

stoplist = stopwords.words('english')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

import requests
import json

url = "http://paraphrase.org/api/en/search/"


def get_synonyms(word):
    results = []
    querystring = {"batchNumber": "0", "needsPOSList": "true", "q": word}
    headers = {
        'cache-control': "no-cache",
        'postman-token': "2d3d31e7-b571-f4ae-d69b-8116ff64d752"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    response_js = response.json()

    res_count = response_js['hits']['found']
    if res_count > 0:
        res_count = min(3, res_count)
        hits = response_js['hits']['hit'][:res_count]
        results = [hit['target'] for hit in hits]
    return results

train = pd.read_csv('input/train_process1.csv').dropna().reset_index(drop=True)
vader = SentimentIntensityAnalyzer()

def sentiment_scores(word):
    score = vader.polarity_scores(word)
    return score

def sort_by_len(lst):
    lst.sort(key=len)
    return lst


def find_synonym(text, selected_text, sentiment):
    selected_texts = []
    texts = []

    orig_words = selected_text.split()
    words = [word_tokenize(str(word)) for word in orig_words]
    words = [word[0] for word in words if len(word) > 0]

    polar_words = []
    if sentiment == 'positive':
        polar_words = [word for word in words if sentiment_scores(word)['pos'] > 0]
    elif sentiment == 'negative':
        polar_words = [word for word in words if sentiment_scores(word)['neg'] > 0]

    if len(polar_words) == 0:
        b = orig_words[0]
        b = re.sub(r'\W+', '', b).lower()
        for word in orig_words:
            if (len(word) > len(b)):
                b = word
        polar_words = [b]

    polar_word = sort_by_len(polar_words)[-1]

    try:
        similar_words = get_synonyms(polar_word)

        for similar in similar_words:
            selected_texts.append(re.sub(polar_word, similar, selected_text))
            texts.append(re.sub(polar_word, similar, text))

    except Exception as e:
        print(e)
        if texts == [] and selected_texts == []:
            return ('', '')

    return (texts, selected_texts)

generated = train.progress_apply(lambda x : find_synonym(x.text,x.selected_text, x.sentiment),axis=1)
x,y = list(map(list,zip(*generated.values.tolist())))

new_df=pd.DataFrame({"textID": train.textID.values,"text":x,"sentiment":train.sentiment.values,'selected_text':y})
new_df.to_csv('input/twitter_augmented.csv',index=False)
print(new_df.head())

