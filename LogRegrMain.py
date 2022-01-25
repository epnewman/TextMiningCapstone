# %load SentModelingScript.py
import pandas as pd
import numpy as np
import spacy
import re
from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from time import time

#declaring cvariables
nlp = spacy.load("en_core_web_sm")
lemmatizer = nlp.vocab.morphology.lemmatizer

#Loading in manually labeled sample reviews
df = pd.read_excel('SampleReviews.xlsx')
df = df[['RowID','reviewTitle', 'reviewText', 'Sentiment Score']]

#Loading in entire dataset
df_full = pd.read_csv('cleanedReview.csv')
df_full = df_full[~df_full['RowID'].isin(df['RowID'])]
df_full = df_full[['RowID','reviewStars','reviewTitle', 'reviewText']]

ros = RandomOverSampler()
vectorizer = TfidfVectorizer(use_idf=True, norm="l2", stop_words="english", max_df=1.0)

#Used for cleaning text
def clean_docs(doc):
    doc = doc.lower() #converts to lower  
    doc = ' '.join(lemmatizer(str(word), VERB)[0] for word in nlp(doc))    #lemmatizes the verbs
    doc = ' '.join(word.text for word in nlp(doc) if not word.is_stop)    #removes stopwords
    doc = ' '.join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", doc).split())    #keeps only alphabets and numbers. change it to [^A-Za-z \t}
    doc = doc.strip()
    return doc 

#Run on initial, entire dataset
def preprocess_entire_data(df):
	df.loc[df.reviewStars <= 3, 'Sentiment Score'] = -1
	df.loc[df.reviewStars >= 4, 'Sentiment Score'] = 1
	df['Sentiment Score'] = df['Sentiment Score'].astype(int)
	df['reviewText'] = df['reviewText'].astype(str)
	df['titleText'] = df['reviewTitle'].str.cat(df['reviewText'],sep=" ")
	df["cleaned_text"] = df.titleText.apply(lambda x: clean_docs(str(x)))
	return df

#Format validation set for final output results
def create_validation_set(df):
	df.reviewText = df.reviewText.astype(str)
	df = df.dropna(subset=['Sentiment Score'])
	df = df[df['Sentiment Score'] != '\xa0']
	df['Sentiment Score'] = df['Sentiment Score'].astype(int)

	new_texts = df.reviewText
	X_new = vectorizer.transform(new_texts)

	return(X_new)

def train_test(X_train, X_test, y_train, y_test, classifier, ismultilabel=False):
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    
    return classifier
    print("Test score: {:.2f}\n".format(classifier.score(X_test, y_test)))   

#Declaring dataset
df_full = preprocess_entire_data(df_full)

#Created vectorization for train and test
X = vectorizer.fit_transform(df_full['cleaned_text'].values.astype('U'))
y = df_full['Sentiment Score']

#Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#Over sampling on the training data
X_ros, y_ros = ros.fit_resample(X_train, y_train)

X_train = X_ros
y_train = y_ros

#run model
lr = LogisticRegression(solver="lbfgs", max_iter = 1000)
lr = train_test(X_train, X_test, y_train, y_test, lr)

#create validation set, based on labeled data
X_new = create_validation_set(df)

#predict
predictions = lr.predict(X_new)
predictions = list(predictions)

#output results
df['Predicted Sentiment'] = predictions

print("Script Executed, Results outputted to csv file")
df.to_csv('validationResults.csv')