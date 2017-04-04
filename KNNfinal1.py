                                                                                                                                                                                                                                                              #Leave-One-Subject-Out
import csv
import random
from random import randrange
import numpy 
import math

import operator

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
		predictedvalue,expectedvalue=algorithm(train_set,test_set,5)
		predicted.append(predictedvalue)
		expected.append(expectedvalue)
	print "Predicted",predicted
	print "EXpected" ,expected
	accuracy_metric(expected,predicted)
	
def euclideanDistance(instance1,instance2,length):
	distance = 0
	
	for x in range(length):
		a=(instance1[x])
		b=(instance2[x])
		distance+=pow((a-b),2)
		
	return math.sqrt(distance)
 

def algorithm(train_set,test_set,k1):
	distances=[]
	predicted=list()
	for i in range(len(test_set)):
		length=len(test_set[i])-1
		for k in range(len(train_set)):
			dist=euclideanDistance(test_set[i],train_set[k],length)
			distances.append((train_set[k],dist))
		distances.sort(key=operator.itemgetter(1))
		neighbors=[]
		for x in range(k1):
			neighbors.append(distances[x][0])
		predicted=getResponse(neighbors)

	for i in range(len(test_set)):
		class1=0
		class0=0
		response1=test_set[i][-1]
		if(response1==1):
			class1=class1+1
		else:
			class0=class0+1
		if(class1>class0):
			expectedvote=1
		else:
			expectedvote=0
	return predicted,expectedvote


def getResponse(neighbors):
	class1=0
	class0=0
	for x in range(len(neighbors)):
		response=neighbors[x][-1]
		if(response==1):
			class1=class1+1
		else:
			class0=class0+1
	if(class1>class0):
		selectedvote=1
	else:
		selectedvote=0
	return selectedvote


def main():
	filename="train_data2.csv"
	dataset=load_csv(filename)
	evaluate_algorithm(dataset,40)
	#print('Accuracy: %.3f%%' % (sum(accuracyscores)/float(len(accuracyscores))*100))
	#print('MCC: %.3f%%' % (sum(mccscores)/float(len(mccscores))))
	#print('Specificity: %.3f%%' % (sum(Specificityscores)/float(len(Specificityscores))))
	#print('Sensitivity: %.3f%%' % (sum(Sensitivityscores)/float(len(Sensitivityscores))))

main()

