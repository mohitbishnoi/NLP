# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:31:36 2019

@author: m
"""
import nltk
nltk.download()

#Tokenize words
from nltk.tokenize import sent_tokenize, word_tokenize
data = "All work and no play makes jack a dull boy, all work and no play"
print(word_tokenize(data))

#Tokenizing sentences
#data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
print(sent_tokenize(data))

#NLTK stop words
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
 
data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
stopWords = set(stopwords.words('english'))
words = word_tokenize(data)
wordsFiltered = []
print(words)
for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)
 
print(wordsFiltered)

#We get a set of English stop words using the line:

stopWords = set(stopwords.words('english'))
#stopWords contains 179 stop words
print(len(stopWords))
print(stopWords)

#NLTK – stemming
#A word stem is part of a word. It is sort of a normalization idea, but linguistic.
#For example, the stem of the word waiting is wait.

words = ["game","gaming","gamed","games"]

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

words = ["game","gaming","gamed","games"]
ps = PorterStemmer()
 
for word in words:
    print(ps.stem(word))

#NLTK speech tagging
#Given a sentence or paragraph, it can label words such as verbs, nouns and so on    

import nltk
from nltk.tokenize import PunktSentenceTokenizer
 
document = 'Whether you\'re new to programming or an experienced developer, it\'s easy to learn and use Python.'
sentences = nltk.sent_tokenize(document)   
for sent in sentences:
    print(nltk.pos_tag(nltk.word_tokenize(sent)))    
    
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
 
document = 'Today the Netherlands celebrates King\'s Day. To honor this tradition, the Dutch embassy in San Francisco invited me to'
sentences = nltk.sent_tokenize(document)   
 
data = []
for sent in sentences:
    data = data + nltk.pos_tag(nltk.word_tokenize(sent))
 
for word in data: 
    if 'NNP' in word[1]: 
        print(word)    
        

# =============================================================================
# nlp prediction example
# Given a name, the classifier will predict if it’s a male or female.
# 
# To create our analysis program, we have several steps:
# 
# Data preparation
# Feature extraction
# Training
# Prediction        
# =============================================================================

from nltk.corpus import names
 
# Load data and training 
names = ([(name, 'male') for name in names.words('male.txt')] + 
	 [(name, 'female') for name in names.words('female.txt')])

def gender_features(word): 
    return {'last_letter': word[-1]}

featuresets = [(gender_features(n), g) for (n,g) in names]
train_set = featuresets
classifier = nltk.NaiveBayesClassifier.train(train_set) 
 
# Predict
print(classifier.classify(gender_features('rockbottom')))

name = input("Name: ")
print(classifier.classify(gender_features(name)))