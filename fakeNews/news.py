import pandas as pd
import numpy as np
import re
import itertools
import random
import join
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def loadData():
	global text_train
	global text_test
	global X_train
	global X_test
	global X_train_tfidf
	global X_test_tfidf
	global Y_train
	global Y_test
	join.main()
	print('Loading data...')
	df_train = pd.read_csv('train.csv', header=None, encoding='ISO-8859-1')
	df_test = pd.read_csv('test.csv', header=None, encoding='ISO-8859-1')
	'''
	#open csv and extract training and testing data
	df_train = df.loc[df[0] == 'train']
	df_test = df.loc[df[0] == 'test']
	'''
	#extract the training and testing data into lists containing article headlines and article bodies
	Y_train = df_train.iloc[0:, 0].values
	text_train = df_train.iloc[0:, 1].values + " " + df_train.iloc[0:, 2].values
	Y_test = df_test.iloc[0:, 0].values
	text_test = df_test.iloc[0:, 1].values + " " + df_test.iloc[0:, 2].values

	#use countVectorizer to create certain ngrams of the words in the test and train data
	vect = CountVectorizer(stop_words='english', ngram_range=(1,4))
	X_train = vect.fit_transform(text_train)
	X_train.shape
	#use tfidfTransformer to create a tf-idf representation of the training and testing data
	tfidf = TfidfTransformer()
	X_train_tfidf = tfidf.fit_transform(X_train)
	X_train_tfidf.shape

	X_test = vect.transform(text_test)
	X_test.shape
	X_test_tfidf = tfidf.fit_transform(X_test)
	X_test_tfidf.shape
	print('...done')
	'''
	####IMPLEMENTATION OF CAPTIONED DATA (TRAIN ONLY)####
	text_train_head = "HEADLINE " + df_train.iloc[0:, 2].values
	text_train_body = "BODY " + df_train.iloc[0:, 3].values

	text_test_head = "HEADLINE" + df_test.iloc[0:,2].values
	text_test_body = "BODY" + df_test.iloc[0:, 3].values

	vect = CountVectorizer(stop_words='english', ngram_range=(1,3))
	X_train_head = vect.fit_transform(text_train_head)
	X_train_body = vect.fit_transform(text_train_body)
	X_train_head.shape
	X_train_body.shape
	X_train = np.hstack((X_train_head, X_train_body))

	X_test_head = vect.fit_transform(text_test_head)
	X_test_body = vect.fit_transform(text_test_body)
	X_test_head.shape
	X_test_body.shape
	X_test = np.hstack((X_test_head, X_test_body))
	'''
	
	print('Number of training instances: %d' % len(df_train))
	print('Number of testing instances: %d' % len(df_test))

def doPtr():
	choice = raw_input('Would you like to do validation scoring on Perceptron? It will take a while. (y or n) ')
	if(choice == 'y'):
		numuse = input('Enter number of instances you want to use: ')
		cross = input('Enter number of folds for cross validation: ')
		#'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
		params = [{'penalty': ['l1', 'l2', 'none', 'elasticnet'], 'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]}]
		print('Fitting Perceptron...')
		PerClassifier = SGDClassifier(loss='perceptron', max_iter=100000, tol=1e-12, random_state=123, average=True).fit(X_train[:numuse], Y_train[:numuse])
		print('...done')
		print('Running grid search and scoring...')
		gs_classifier = GridSearchCV(PerClassifier, params, cv=cross, n_jobs=-1)
		gs_classifier.fit(X_train_tfidf[:numuse], Y_train[:numuse])
		print('Best parameter(Perceptron): %s' % gs_classifier.best_params_)
		print('Validation Accuracy(Perceptron): %0.6f' % gs_classifier.best_score_)
		print('Training Accuracy(Perceptron): %0.6f' % PerClassifier.score(X_train_tfidf[:numuse], Y_train[:numuse]))
		print('Test Accuracy(Perceptron): %0.6f\n' % PerClassifier.score(X_test_tfidf, Y_test))
		choice = raw_input('Would you like to see the confusion matrix? (y or n): ')
		if(choice == 'y'):
			label_names = ['agree', 'disgree', 'discusses', 'unrelated']
			print(confusion_matrix(Y_test, PerClassifier.predict(X_test_tfidf), label_names))
	else:
		print('Fitting Perceptron...')
		PerClassifier = SGDClassifier(loss='perceptron', max_iter=10000, tol=1e-12, random_state=123, average=True).fit(X_train_tfidf, Y_train)
		print('Done...')
		print('Scoring...')
		print('Training Accuracy: %0.6f' % accuracy_score(Y_train, PerClassifier.predict(X_train_tfidf)))
		print('Testing Accuracy: %0.6f' % accuracy_score(Y_test, PerClassifier.predict(X_test_tfidf)))
		print('Done...')

def doLR():
	choice = raw_input('Would you like to do validation scoring on Logistic Regression Classifier? It will take a while. (y or n) ')
	if(choice == 'y'):
		numuse = input('Enter number of instances you want to use: ')
		cross = input('Enter number of folds for cross validation: ')
		params = [{'penalty': ['l1', 'l2'], 'dual': [True, False], 'C': np.arange(0.1, 2.0 , 0.1)}]
		print('Fitting Logistic Regression Classifier...')
		LRClassifier = LogisticRegression(multi_class='multinomial', solver = 'lbfgs', random_state=123).fit(X_train_tfidf, Y_train)
		print('...done')
		gs_classifier = GridSearchCV(LRClassifier, params, cv=cross, n_jobs=-1).fit(X_train_tfidf[:numuse], Y_train[:numuse])
		print('Info for Logistic Regression Classifier')
		print('Best parameters: %s' % gs_classifier.best_params_)
		print('Validation Accuracy: %s' % gs_classifier.best_score_)
		print('Training Accuracy: %s' % LRClassifier.score(X_train_tfidf[:numuse], Y_train[:numuse]))
		print('Test Accuracy: %s' % LRClassifier.score(X_test_tfidf, Y_test))
	else:
		print('Fitting Logistic Regression Classifier...')
		LRClassifier = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', random_state = 123).fit(X_train_tfidf, Y_train)
		print('...Done')
		print('Scoring...')
		print('Training Accuracy: %0.6f' % accuracy_score(Y_train, LRClassifier.predict(X_train_tfidf)))
		print('Testing Accuracy: %0.6f' % accuracy_score(Y_test, LRClassifier.predict(X_test_tfidf)))


def doNB():
	choice = raw_input('Would you like to do validation scoring on Naive Bayes classifier? It will take a while. (y or n) ')
	if(choice == 'y'):
		cross = input('Enter number of folds for cross validation: ')
	#try different alpha,fit_prior and n-grams to find the best combination with the given data
		params = [{'alpha': np.arange(0.1, 2.0, 0.1), 'fit_prior': [True, False]}]
		print('Fitting Naive Bayes classifier...')
		nb_classifier = MultinomialNB().fit(X_train_tfidf, Y_train)
		print('...done\n')
		print('Running a grid search and scoring...')
		gs_classifier = GridSearchCV(nb_classifier, params, cv=cross, n_jobs=-1)
		gs_classifier.fit(X_train_tfidf, Y_train)
		print('Best parameter(Naive Bayes): %s' % gs_classifier.best_params_)
		print('Validation Accuracy(Naive Bayes): %0.6f' % gs_classifier.best_score_)
		print('Training Accuracy(Naive Bayes): %0.6f' % nb_classifier.score(X_train_tfidf, Y_train))
		print('Test Accuracy(Naive Bayes): %0.6f\n' % nb_classifier.score(X_test_tfidf, Y_test))
		choice = raw_input('Would you like to see the confusion matrix? (y or n): ')
		if(choice == 'y'):
			label_names = ['agree', 'disgree', 'discusses', 'unrelated']
			print(confusion_matrix(Y_test, nb_classifier.predict(X_test_tfidf), label_names))
	else:
		#implement Naive Bayes classifier
		#best parameters: alpha=1.5 fit_prior = False
		print('Fitting Naive Bayes classifier...')
		nb_classifier = MultinomialNB(alpha=1.5, fit_prior=False).fit(X_train_tfidf, Y_train)
		print('...done\n')
		print('Training Accuracy(Naive Bayes): %0.6f' % nb_classifier.score(X_train_tfidf, Y_train))
		print('Test Accuracy(Naive Bayes): %0.6f\n' % nb_classifier.score(X_test_tfidf, Y_test))
		choice = raw_input('Would you like to see the confusion matrix? (y or n): ')
		if(choice == 'y'):
			label_names = ['agree', 'disgree', 'discusses', 'unrelated']
			print(confusion_matrix(Y_test, nb_classifier.predict(X_test_tfidf)))



def doSGD():
	choice = raw_input('Would you like to do validation scoring on SGD Classifier? It will take a while. (y or n) ')
	if(choice == 'y'):
		cross = input('Enter number of folds for cross validation: ')
		print('Fitting SGD Classifier...')
		sgd_classifier = SGDClassifier(tol=1e-3, max_iter=10000, random_state=123).fit(X_train_tfidf, Y_train)
		print('...done\n')
		params = [{'penalty': ['l1', 'l2', 'none', 'elasticnet'], 'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], 'loss': ['hinge', 'modified_huber', 'squared_hinge', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']}]

		print('Running a grid search and scoring...')
		gs_classifier = GridSearchCV(sgd_classifier, params, cv=cross, n_jobs=-1)
		gs_classifier.fit(X_train_tfidf, Y_train)

		print('Scoring...')
		print('Best parameter(SGD): %s' % gs_classifier.best_params_)
		print('Validation Accuracy: %0.6f\n' % gs_classifier.best_score_)
		print('Training Accuracy(SGD): %0.6f' % sgd_classifier.score(X_train_tfidf, Y_train))
		print('Testing Accuracy: %0.6f\n' % sgd_classifier.score(X_test_tfidf, Y_test))
		choice = raw_input('Would you like to see the confusion matrix? (y or n): ')
		if(choice == 'y'):
			label_names = ['agree', 'disgree', 'discusses', 'unrelated']
			print(confusion_matrix(Y_test, sgd_classifier.predict(X_test_tfidf)))
	else:
		#implement SGD Classifier
		#best parameters: penalty = none, alpha = 1e-5,loss = hinge
		print('Fitting SGD Classifier...')
		sgd_classifier = SGDClassifier(penalty='none', alpha=1e-5, loss='hinge', tol=1e-3, max_iter = 100000, random_state=123).fit(X_train_tfidf, Y_train)
		print('...done\n')
		print('Training Accuracy(SGD): %0.6f' % sgd_classifier.score(X_train_tfidf, Y_train))
		print('Testing Accuracy: %0.6f\n' % sgd_classifier.score(X_test_tfidf, Y_test))
		choice = raw_input('Would you like to see the confusion matrix? (y or n): ')
		if(choice == 'y'):
			label_names = ['agree', 'disgree', 'discusses', 'unrelated']
			print(confusion_matrix(Y_test, sgd_classifier.predict(X_test_tfidf)))


def doSVM():
	numtrain = input('Number of instances to use for training(SVM): ')
	choice = raw_input('Would you like to do validation scoring on SVM Classifier? It will take a while. (y or n) ')
	if(choice == 'y'):
		sv_classifier = SVC(kernel='linear', random_state=123).fit(X_train_tfidf[:numtrain], Y_train[:numtrain])
		cross = input('Enter number of folds for cross validation: ')
		#{'gamma': gamma_range, 'C' : np.arange(0.1, 2.0 , 0.1)}
		gamma_range = np.logspace(-9,3,13)
		params = [{'gamma': gamma_range, 'C' : np.arange(0.1, 2.0 , 0.1)}]
		print('Running a grid search and scoring...')
		gs_classifier = GridSearchCV(sv_classifier, params, cv=cross, n_jobs=-1)
		gs_classifier.fit(X_train_tfidf[:numtrain], Y_train[:numtrain])
		print('Scoring...')
		print('Best parameter(SVM): %s' % gs_classifier.best_params_)
		print('Validation Accuracy(SVM): %0.6f\n' % gs_classifier.best_score_)
		print('Training Accuracy(SVM): %0.6f' % sv_classifier.score(X_train_tfidf[:numtrain], Y_train[:numtrain]))
		print('Test Accuracy(SVM): %0.6f\n' % sv_classifier.score(X_test_tfidf[:numtrain], Y_test[:numtrain]))
		choice = raw_input('Would you like to see the confusion matrix? (y or n): ')
		if(choice == 'y'):
			label_names = ['agree', 'disgree', 'discusses', 'unrelated']
			print(confusion_matrix(Y_test, sv_classifier.predict(X_test_tfidf)))
	else:
		#implement SVM classifier1
		#best parameters: C = 0.4, gamma = 1e-9
		print('Fitting SVM classifier...')
		sv_classifier = SVC(C=0.6, gamma=1e-7, kernel='linear', random_state=123).fit(X_train_tfidf[:numtrain], Y_train[:numtrain])
		print('...done\n')
		print('Training Accuracy(SVM): %0.6f' % sv_classifier.score(X_train_tfidf[:numtrain], Y_train[:numtrain]))
		print('Test Accuracy(SVM): %0.6f\n' % sv_classifier.score(X_test_tfidf[:numtrain], Y_test[:numtrain]))
		choice = raw_input('Would you like to see the confusion matrix? (y or n): ')
		if(choice == 'y'):
			label_names = ['agree', 'disgree', 'discusses', 'unrelated']
			print(confusion_matrix(Y_test, sv_classifier.predict(X_test_tfidf)))

def doNN():
	print('Fitting classifier...')
	nnclassifier = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes = (20,10), alpha=1e-5, random_state=123).fit(X_train_tfidf[:10000], Y_train[:10000])
	print('...done')
	print('Calculating accuracies...')
	print('Training Accuracy: %0.6f' % nnclassifier.score(X_train_tfidf, Y_train))
	print('Test Acccuracy: %0.6f' % nnclassifier.score(X_test_tfidf, Y_test))


def main():
	done=False
	firstTime=-1
	while(not done):
		choice = raw_input('Would you like to classify (y or n): ')
		if(choice=='n'):
			print('Goodbye!')
			done=True
			break
		firstTime+=1
		if(not firstTime):
			loadData()
		print('\nList of different classifiers for Fake News Challenge: ')
		print('1. Perceptron\n2. Logistic Regression\n3. Naive Bayes\n4. Stochastic Gradient Descent\n5. Support Vector Machine\n6. Neural Network\n7. All the above (not ensembled)')
		classchoice = input('Enter a number for the classifier you would like to use: ')
		if(classchoice == 1):
			doPtr()
		elif(classchoice == 2):
			doLR()
		elif(classchoice == 3):
			doNB()
		elif(classchoice == 4):
			doSGD()
		elif(classchoice == 5):
			doSVM()
		elif(classchoice == 6):
			doNN()
		elif(classchoice == 7):
			doPtr()
			doLR()
			doNB()
			doSGD()
			doSVM()
			doNN()
		else:
			print('You entered the wrong number...you suck')

if __name__ == '__main__':
	main()