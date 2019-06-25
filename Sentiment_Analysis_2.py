# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:39:46 2019

@author: Megha
"""


# Import packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
#importing TfidfVectorizer using sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

#Loading Data
data=pd.read_csv("F:/Python/Python Class notes/NLP/Case Study Sentiment Analysis/sentiment-analysis-on-movie-reviews/train.tsv", sep='\t')
data.head()

data.info()
#This data has 5 sentiment labels:
#0 - negative 1 - somewhat negative 2 - neutral 3 - somewhat positive 4 - positive

data.Sentiment.value_counts()

Sentiment_count=data.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()

#data["Sentiment"] = data["Sentiment"].replace(0, 0).replace(1, 0).replace(2, 0)
#data["Sentiment"] = data["Sentiment"].replace(3, 1).replace(4, 1)

#data.Sentiment.value_counts()

Sentiment_count=data.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()

#Feature Generation using Bag of Words
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,2),tokenizer = token.tokenize)
text_counts= cv.fit_transform(data['Phrase'])
print(text_counts)
#Split train and test set
X_train, X_test, y_train, y_test = train_test_split(text_counts, data['Sentiment'], test_size=0.1, random_state=10)

 #Model Building and Evaluation
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)

predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


#got a classification rate of 60.49% using CountVector(or BoW), which is not considered as good accuracy.

#Feature Generation using TF-IDF

tf=TfidfVectorizer()
text_tf= tf.fit_transform(data['Phrase'])
#print(text_tf)
#Split train and test set (TF-IDF)
X_train, X_test, y_train, y_test = train_test_split(text_tf, data['Sentiment'], test_size=0.2, random_state=10)

#Model Building and Evaluation (TF-IDF)
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
#Accuracy: 0.587







 


