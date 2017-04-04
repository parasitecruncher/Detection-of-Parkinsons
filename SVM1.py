                                                                                                                                                                                                                                                              #Leave-One-Subject-Outimport csvimport random
from random import randrange
import numpy 
import math
import csv
from sklearn import metrics
from sklearn.svm import SVC
import operator
import copy

def load_csv(filename):
	with open(filename,'r') as file:
		reader=csv.reader(file)
		dataset=list(reader)
		total=len(dataset)
		print total
		for i in range(0,total):
			for value in dataset[i]:
				value=float(value)
		for i in range(0,total):
			dataset[i] = map(float,dataset[i])
		return dataset


def loso(dataset,numberofsubjects):
		dataset_split=list()
		dataset_copy=list(dataset)
		total=len(dataset_copy)
		for i in range(0,total):
			dataset_copy[i].pop(0)
		fold_size=int(len(dataset)/numberofsubjects)
		for i in range(numberofsubjects):
		
			fold=list()
		#while len(fold)<fold_size:
			for  i in range(fold_size):
				fold.append(dataset_copy.pop(0))
			dataset_split.append(fold)
		
		return dataset_split


def accuracy_metric(expected, predicted):
	truepositive = 0
	truenegative = 0
	falsepositive= 0
	falsenegative = 0
	#print expected
	#print predicted
	for i in range(len(expected)):
		if (expected[i]== 1):
			if (predicted[i]==1):
				truepositive+=1
			else:
				falsenegative+=1
		elif (expected[i]==0):
			if (predicted[i]==0):
				truenegative+=1
			else:
				falsepositive+=1
	print truepositive
	print truenegative
	print falsepositive
	print falsenegative

	accuracy=(1.0*(truenegative+truepositive))/(truenegative+truepositive+falsenegative+falsepositive)*100
	print "Accuracy :",accuracy,"%"
	MCC=(1.0*((truepositive*truenegative)-(falsepositive*falsenegative)))/(math.sqrt((truepositive+falsepositive)*(truepositive+falsenegative)*(truenegative+falsepositive)*(truenegative+falsenegative)))
	print "MCC:",MCC
	Specificity=(1.0*truenegative)/(truenegative+falsepositive)*100
	print "Specificity:",Specificity,"%"
	Sensitivity=(1.0*truepositive)/(truepositive+falsenegative)*100 
	print "Sensitivity",Sensitivity,"%"

def evaluate_algorithm(dataset,numberofsubjects):
	iteration=0
	predicted=list()
	expected=list()
	folds=loso(dataset,numberofsubjects)
	for fold in folds:
		train_set=list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy=list(row)
			test_set.append(row_copy)
		#print "Iteration",iteration
		predictedvalue,expectedvalue=algorithm(train_set,test_set)
		predicted.append(predictedvalue)
		expected.append(expectedvalue)
	print "Expected", expected
	print "Predicted", predicted
	accuracy_metric(expected,predicted)

def algorithm(train_set,test_set):
	copy_of_test_set=copy.deepcopy(test_set)
	copy_of_train_set=copy.deepcopy(train_set)
	test_setclass=list()
	train_setclass=list()
	for i in range(26):
		test_setclass.append(copy_of_test_set[i][-1])
		del copy_of_test_set[i][-1]
	for k in range(1014):
		train_setclass.append(copy_of_train_set[k][-1])
		del copy_of_train_set[k][-1]
	#model = SVC(C=10.0,kernel='rbf',gamma=0.0003)
	model = SVC(kernel='linear')
	model.fit(copy_of_train_set,train_setclass)
	expected = test_setclass
	predicted = model.predict(copy_of_test_set)
	predictedvalue=getResponse(predicted)

	class1=0
	class0=0
	for i in range(len(test_set)):
		response1=test_set[i][-1]
		if(response1==1):
			class1=class1+1
		else:
			class0=class0+1
		if(class1>class0):
			expectedvote=1
		else:
			expectedvote=0
	return predictedvalue,expectedvote


def getResponse(predicted):
	class1=0
	class0=0
	for x in range(len(predicted)):
		response=predicted[x]
		if(response==1):
			class1=class1+1
		else:
			class0=class0+1
	#print "Class1 predicted",class1
	#print "Class0 predicted",class0
	if(class1>class0):
		selectedvote=1
	else:
		selectedvote=0
	return selectedvote


def main():
	filename="train_data2.csv"
	dataset=load_csv(filename)
	evaluate_algorithm(dataset,40)
	
main()

