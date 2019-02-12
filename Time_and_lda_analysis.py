import pandas as pd
import datetime as dt
from nltk import bigrams,trigrams
from gensim.corpora.dictionary import Dictionary
from gensim import  models
import re
pd.set_option('display.max_colwidth', -1)

#Parameters
# value between 0.1 to 1. filter days when there were fewer tweets than this quantile
quantile_filter = 0.5
num_topics=2
filter_words_that_appeared_less_than =5
use_bigrams_phrase =True
use_trigrams_phrase = True
remove_user_name=True

#load tweets
tweets = pd.read_csv('tweets_israeli-girls.csv')
#set 'Text column as text
tweets['Text'] = tweets['Text'].astype(str)
#set text column as datetime
tweets['Date Created'] = pd.to_datetime(tweets['Date Created'])

#add column only the hour of 'Date Created'
hours = []
for d in tweets['Date Created']:
    hours.append(d.hour)
tweets['Hour'] = hours

# find tweets min and max 'Date Created'
max_date =  tweets['Date Created'].max()
max_date_limit = dt.datetime(max_date.year,max_date.month,max_date.day)+dt.timedelta(days=1)
min_date = tweets['Date Created'].min()
min_date_limit = dt.datetime(min_date.year,min_date.month,min_date.day)

dic_tweet_per_day = dict()
start_date =min_date_limit
while (start_date<max_date_limit):
    end_date = start_date+dt.timedelta(days=1)
    current = tweets[tweets['Date Created'].between(start_date,end_date)]
    if len(current)>0:
        dic_tweet_per_day[start_date] =current
    start_date = end_date

dic_counts = {'Date':[], 'Count':[]}
for k in dic_tweet_per_day.keys():
    dic_counts['Date'].append(k)
    count =len(dic_tweet_per_day[k])
    dic_counts['Count'].append(count)

count_df = pd.DataFrame(dic_counts)
filter_value = count_df['Count'].quantile(quantile_filter)

with open("./stopwords.txt", encoding="utf-8") as f:
    stopwords = set(f.readlines())
    stopwords = {w.replace("\n", "") for w in stopwords}

def preprocess(input_str):
    PUNCTUATION = {",", ".", "!", "?", ";", "-", "*", "&", "|",":" , '(', ')'}
    result = input_str
    #remove links
    result = " ".join([t for t in result.split() if not t.startswith("http")])
    mentions = re.findall("@[a-zA-Z0-9_.]*", result)
    #remove user name (@UserName)
    if remove_user_name:
        for mention in mentions:
            result = result.replace(mention, "")
    #remove punctuation
    for sign in PUNCTUATION:
        result = result.replace(sign, " ")
    #remove '
    result = result.replace("'", "")
    #remove stopwords
    for stopword in stopwords:
        result = result.replace(" {}".format(stopword), "")
        result = result.replace("{} ".format(stopword), "")
    #create unigram, bigrams and trigrams
    unigram = [w for w in result.split() if len(w)>1]
    features = unigram
    if use_bigrams_phrase:
        bigrams_phrase = [b[0]+" "+b[1] for b in bigrams(unigram)]
        features += bigrams_phrase
    if use_trigrams_phrase:
        trigrams_phrase =[b[0]+" "+b[1]+" "+b[2] for b in trigrams(unigram)]
        features += trigrams_phrase
    return features


dic_for_data_fram = {'Date': [], 'Words': []}

dic_suspicion = dict()
start_date = min_date_limit
while (start_date < max_date_limit):
    end_date = start_date + dt.timedelta(days=1)
    current = tweets[tweets['Date Created'].between(start_date, end_date)]
    if len(current) >= filter_value:
        df = current

        processed_docs = df['Text'].map(preprocess)

        dictionary = Dictionary(processed_docs)
        dictionary.filter_extremes(no_below=filter_words_that_appeared_less_than, keep_n=100000)

        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        if len(dictionary) > 0:
            print(start_date)
            lda_model = models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=2)

            for idx, topic in lda_model.print_topics(-1):
                print(topic)
                dic_for_data_fram['Date'].append(start_date)
                dic_for_data_fram['Words'].append(topic)

    dic_tweet_per_day[start_date] = current
    start_date = end_date

pd.DataFrame(dic_for_data_fram).to_csv('./suspicion_bots_topics.csv',index=False)

