import pandas as pd
import numpy as np
import re
import itertools
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

df = pd.read_csv('train.csv', header=None, encoding='ISO-8859-1')
#open csv and extract training and testing data
df_train = df.loc[df[0] == 'train']
df_test = df.loc[df[0] == 'test']
#print df_train

Y_train = df_train.iloc[0:, 1].values
#text_train = "HEADLINE " + df_train.iloc[0:, 2].values + " BODY " + df_train.iloc[0:, 3].values
text_train = df_train.iloc[0:, 2].values + " " + df_train.iloc[0:, 3].values

Y_test = df_test.iloc[0:, 1].values
text_test = "HEADLINE " + df_test.iloc[0:, 2].values + " BODY " + df_test.iloc[0:, 3].values
text_test = df_test.iloc[0:, 2].values + " " + df_test.iloc[0:, 3].values
#use countVectorizer to create certain ngrams of the words in the test and train data
vect = CountVectorizer(stop_words='english', ngram_range=(1,3))
X_train = vect.fit_transform(text_train)
X_train.shape
#use tfidfTransformer to create a tf-idf representation of the training and testing data
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_train_tfidf.shape

X_test = vect.transform(text_test)
X_test.shape
tfidftest = TfidfTransformer()
X_test_tfidf = tfidftest.fit_transform(X_test)
X_test_tfidf.shape

#try different alpha,fit_prior and n-grams to find the best combination with the given data
paramsnb = [{'alpha': np.arange(0.1, 2.0, 0.1)}, {'fit_prior': [True, False]}]
paramssvm = [{'C' : np.arange(0.1, 2.0 , 0.1)}, {'kernel' : ['linear', 'rbf', 'poly', 'sigmoid']}]
#implement Naive Bayes classifier and fit it on training data
print('Fitting Naive Bayes classifier...')
nb_classifier = MultinomialNB().fit(X_train_tfidf, Y_train)
print('...done')
print('Fitting SVM classifier...')
sv_classifier = SVC(random_state=123).fit(X_train_tfidf, Y_train)
print('...done')
#try different alpha,fit_prior and n-grams to find the best combination with the given data
#params = [{'tfidf__use_idf': [True, False]},{'vect__ngram_range': [(1,1), (1,2)]}, {'base_classifier1__alpha': np.arange(0.1, 2.0, 0.1)}, {'base_classifier1__fit_prior': [True, False]}]
gs_classifier = GridSearchCV(nb_classifier, paramsnb, cv=7, n_jobs=-1)
gs_classifier.fit(X_train_tfidf, Y_train)

print('Best parameter(Naive Bayes): %s' % gs_classifier.best_params_)
print('Train Accuracy(Naive Bayes): %0.6f' % nb_classifier.score(X_train_tfidf, Y_train))
print('Validation Accuracy(Naive Bayes): %0.6f' % gs_classifier.best_score_)
print('Test Accuracy(Naive Bayes): %0.6f' % nb_classifier.score(X_test_tfidf, Y_test))

gs_classifier = GridSearchCV(sv_classifier, paramssvm, cv = 5, n_jobs=-1)
gs_classifier.fit(X_train_tfidf, Y_train)

print('Best parameter(SVM): %s' % gs_classifier.best_params_)
print('Train Accuracy(SVM): %0.6f' % sv_classifier.score(X_train_tfidf, Y_train))
print('Validation Accuracy(SVM): %0.6f' % gs_classifier.best_score_)
print('Test Accuracy(SVM): %0.6f' % sv_classifier.score(X_test_tfidf, Y_test))