import pandas as pd
import numpy as np
import join
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

def getData():

	# label: 1 is Malignant 0 is benign
	global X_train
	global Y_train
	global X_test
	global Y_test
	print('Loading Data...')
	join.main()
	df_train = pd.read_csv('train.csv', header=None, encoding='ISO-8859-1')
	df_test = pd.read_csv('test.csv', header=None, encoding='ISO-8859-1')
	Y_train = df_train.iloc[0:, 0].values
	X_train = df_train.iloc[0:, 1:].values
	Y_test = df_test.iloc[0:,0].values
	X_test = df_test.iloc[0:,1:].values
	print('Number of training instances: %d' % len(df_train))
	print('Number of testing instances: %d' % len(df_test))

def doPtr():
	cross = input('Enter number of folds for cross validation: ')
	#'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
	params = [{'penalty': ['l1', 'l2', 'none', 'elasticnet'], 'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], 'average': [True, False]}]
	classifier = SGDClassifier(loss='perceptron', max_iter=100000, tol=1e-12, random_state=123)
	print('Running grid search and scoring...')
	gs_classifier = GridSearchCV(classifier, params, cv=cross, n_jobs=-1).fit(X_train,Y_train)
	print('Info for Perceptron')
	print('Best parameter(Perceptron): %s' % gs_classifier.best_params_)
	print('Validation Accuracy(Perceptron): %0.6f' % gs_classifier.best_score_)
	print('Test Accuracy(Perceptron): %0.6f\n' % gs_classifier.score(X_test, Y_test))
	choice = raw_input('Would you like to see the confusion matrix? (y or n): ')
	if(choice == 'y'):
		label_names = [1,0]
		print(confusion_matrix(Y_test, gs_classifier.predict(X_test), label_names))

def doLR():
	cross = input('Enter number of folds for cross validation: ')
	params = [{'C': np.arange(0.1, 2.0 , 0.1)}]
	classifier = LogisticRegression(multi_class='multinomial', solver = 'lbfgs', penalty='l2', dual=False, random_state=123)
	print('Running grid search and scoring...')
	gs_classifier = GridSearchCV(classifier, params, cv=cross, n_jobs=-1).fit(X_train, Y_train)
	print('Info for Logistic Regression Classifier')
	print('Best parameters: %s' % gs_classifier.best_params_)
	print('Validation Accuracy: %s' % gs_classifier.best_score_)
	print('Test Accuracy: %s' % gs_classifier.score(X_test, Y_test))
	choice = raw_input('Would you like to see the confusion matrix? (y or n): ')
	if(choice == 'y'):
		label_names = [1,0]
		print(confusion_matrix(Y_test, gs_classifier.predict(X_test), label_names))

def doNN():
	cross = input('Enter number of folds for cross validation: ')
	#(50,), (100,), (150,), 
	params = [{'alpha': [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1], 'learning_rate': ['constant', 'invscaling', 'adaptive'], 'solver': ['adam', 'lbfgs', 'sgd'], 'hidden_layer_sizes' : [(50,), (100,), (150,), (175,), (200,), (300,)]}]
	classifier = MLPClassifier(max_iter=10000, random_state=123)
	print('Running grid search and scoring...')
	gs_classifier = GridSearchCV(classifier,params,cv=cross,n_jobs=-1).fit(X_train, Y_train)
	print('Info for Multilayer Perceptron: ')
	print('Best paramters: %s' % gs_classifier.best_params_)
	print('Validation Accuracy: %s' % gs_classifier.best_score_)
	print('Test Accuracy: %s' % gs_classifier.score(X_test, Y_test))
	choice = raw_input('Would you like to see the confusion matrix? (y or n): ')
	if(choice == 'y'):
		label_names = [1,0]
		print(confusion_matrix(Y_test, gs_classifier.predict(X_test), label_names))

def doAll():
	doPtr()
	doLR()
	doNN()

def main():
	done = False
	frst = -1
	'''
	classes = {
		0: doPtr(),
		1: doLR(),
		2: doNN(),
		3: doAll()
	}
	''' 
	while not done:
		choice = raw_input('Would you like to classify (y or n): ')
		if(choice == 'n'):
			print 'Goodbye!'
			break
		frst += 1
		if(not frst):
			getData()
		print('Classifiers:')
		print('1. Perceptron\n2. Logistic Regression\n3. Multilayer Perceptron\n4. Do all (not ensembled)')
		choice = input('Enter the number of the classifier you would like to use: ')
		if(choice == 1):
			doPtr()
		elif(choice == 2):
			doLR()
		elif(choice == 3):
			doNN()
		elif(choice == 4):
			doAll()
		else:
			print('You entered an invalid number. You will now be redirected')


if __name__ == '__main__':
	main()