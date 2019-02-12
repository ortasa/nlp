# NLP Research and tools

The file is written in Python 3.
It uses the following libraries that require installation: pandas , nltk , gensim. 

The following parameters can be changed:
quantile_filter = 0.5 - value between 0.1 to 1. filtor days when there were fewer tweets than this quantile
num_topics=2 How many topics will be displayed per day
filter_words_that_appeared_less_than =5 A parameter designed to reduce noise by removing rare words
use_bigrams_phrase =True Should the model consider pairs of words
use_trigrams_phrase = True Should the model consider three word combinations
remove_user_name=True Does the model need to remove usernames.
tweets_csv_file_name = 'tweets_israeli-girls.csv'

The output is a file  'suspicion_bots_topics.csv'  which contains dates and topics. 
