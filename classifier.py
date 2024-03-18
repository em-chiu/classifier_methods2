#!/usr/bin/env python
"""create model using scikitlearn classifying ..."""

# import pandas for data handling
import pandas as pd

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
stopwords = stopwords.words('english')
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Libraries for strings
import string
# Regular Expression Library
import re

# Import text vectorizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Import classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Import ML helper function
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

# Import metrics to evaluate model
from sklearn import metrics
from sklearn.metrics import classification_report

# library for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# import data, loading data into dataframe
df = pd.read_csv('dem-vs-rep-tweets.csv')
print(df.shape)  # print shape
df.head()
# Party     0
# Handle    0
# Tweet     0
# dtype: int64

# check nulls and duplicated
print(df.isnull().sum())
print(df.duplicated().sum())
# found 57 duplicates

# remove duplicates, WORK ON
#remove_dupes = [*set(df)]
#print (remove_dupes.duplicated().sum())


# finding class balance
print(df.Party.value_counts())
# Republican    44392
# Democrat      42068
# Name: Party, dtype: int64

# lowercase all words
new_tweets = [tweet.lower() for tweet in df.Tweet]
#print(new_tweets)

# remove all punctuation
new_tweets = [''.join(c for c in s if c not in string.punctuation) for s in new_tweets]
#print(new_tweets)

# remove stop words
def remove_stopwords(data):
    output_array=[]
    for sentence in data:
        temp_list=[]
        for word in sentence.split():
            if word.lower() not in stopwords:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array


output = remove_stopwords(new_tweets)
#print(output)

# build text processing pipeline
def text_pipeline(input_string):
    input_string = make_lower(input_string)
    input_string = remove_punctuation(input_string)
    #input_string = lem_with_pos_tag(input_string)
    input_string = remove_stopwords(input_string)    
    return input_string


df['message_clean'] = df['message']
# df['message_clean'] = df['message_clean'].apply(make_lower)
# df['message_clean'] = df['message_clean'].apply(remove_punctuation)
# df['message_clean'] = df['message_clean'].apply(lem_with_pos_tag)
# df['message_clean'] = df['message_clean'].apply(remove_stopwords)
df['message_clean'] = df['message'].apply(text_pipeline)

print("ORIGINAL TEXT\n:", df['message'][0])
print("CLEANED TEXT\n:", df['message_clean'][0])


# Initialize our vectorizer
vectorizer = TfidfVectorizer()

#  makes vocab matrix
vectorizer.fit(X)

# transforms documents into vectors
X = vectorizer.transform(X)

print(X.shape, type(X))

# Split data into testing and training 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initalize model
model = MultinomialNB(alpha=.05)

# Fit model with training data
model.fit(X_train, y_train)

# Make new predictions of testing data
y_pred = model.predict(X_test)

# Make predicted probabilites of testing data
y_pred_proba = model.predict_proba(X_test)

# Evaluate model
accuracy =  model.score(X_test, y_test)

# Print evaluation metrics
print("Model Accuracy: %f" % accuracy)


