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

# import data, loading data into dataframe # extract data
df = pd.read_csv('dem-vs-rep-tweets_copy.csv') #chunksize, then save to df
print("shape", df.shape)  # print shape to see how data formated
# (86460, 3), tuple representing the dimensionality (rows/columns) of dataframe
print("head", df.head()) # shows beginning of file

# check nulls and duplicates
print("nulls #", df.isnull().sum())
# Party     0
# Handle    0
# Tweet     0
# dtype: int64
print("duplicates #", df.duplicated().sum())
# found 57 duplicates

# remove duplicates, WORK ON
#remove_dupes = [*set(df)] # makes set from dataframe --> unique values only
#remove_dupes = df.drop_duplicates()
#print("removed dupes", remove_dupes.duplicated().sum())
# df = remove_dupes #option to reassign dataframe
# transform/clean data
df.drop_duplicates(inplace=True) # drop dupes & reassign to df
print("removed dupes", df.duplicated().sum())


# finding class balance
print(df.Party.value_counts())
# Republican    44392
# Democrat      42068
# Name: Party, dtype: int64

# lowercase all words
#new_tweets = [tweet.lower() for tweet in df.Tweet]
#print(new_tweets)
def make_lower(a_string):
    return a_string.lower()
# applies function to Pandas column and displays Tweet column in dataframe after applying make_lower function
#print(df.Tweet.apply(make_lower).to_string(index=False))

# remove all punctuation
#new_tweets = [''.join(c for c in s if c not in string.punctuation) for s in new_tweets]
def remove_punctuation(a_string):    
    a_string = re.sub(r'[^\w\s]','',a_string)
    return a_string
# applies function to Pandas column and displays Tweet column in dataframe after applying remove_punctuation function
#print(df.Tweet.apply(remove_punctuation).to_string(index=False))


#output = remove_stopwords(df.Tweet)
#print(output)
def remove_stopwords(a_string):
    # Break the sentence down into a list of words
    words = word_tokenize(a_string)
    # Make a list to append valid words into
    valid_words = []
    # Loop through all the words
    for word in words:
        # Check if word is not in stopwords
        if word not in stopwords:
            # If word not in stopwords, append to our valid_words
            valid_words.append(word)
    # Join the list of words together into a string
    a_string = ' '.join(valid_words)
    return a_string


# build text processing pipeline # transform/cleaning data
def text_pipeline(input_string):
    input_string = make_lower(input_string)
    input_string = remove_punctuation(input_string)
    #input_string = lem_with_pos_tag(input_string)
    input_string = remove_stopwords(input_string)    
    return input_string


# create new column to store cleaned version of tweet
df['message_clean'] = df['Tweet'].apply(text_pipeline)

print("ORIGINAL TEXT\n:", df['Tweet'][0])
print("CLEANED TEXT\n:", df['message_clean'][0])

# # define data
X = df['message_clean'].values

y = df['Tweet'].values

# # Initialize our vectorizer
vectorizer = TfidfVectorizer()

# #  makes vocab matrix
vectorizer.fit(X)

# # transforms documents into vectors
X = vectorizer.transform(X)

print(X.shape, type(X))
# #(86460, 132962) <class 'scipy.sparse.csr.csr_matrix'>

# # Split data into testing and training 
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42)

# # Initalize model # load data (training classifier using cleaned data)
model = MultinomialNB(alpha=.05)

# # Fit model with training data
model.fit(X_train, y_train)

# # Make new predictions of testing data
y_pred = model.predict(X_test)

# # Make predicted probabilites of testing data
y_pred_proba = model.predict_proba(X_test)

# # Evaluate model
accuracy =  model.score(X_test, y_test)

# # Print evaluation metrics
print("Model Accuracy: %f" % accuracy)
# # (86460, 132962) <class 'scipy.sparse.csr.csr_matrix'>