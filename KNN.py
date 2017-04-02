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


def accuracy_metric(test_set, predicted):
	correct = 0
	for i in range(len(test_set)):
		if test_set[i][-1] == predicted:
			correct += 1
	return correct / float(len(test_set)) * 100.0   


def evaluate_algorithm(dataset,numberofsubjects):
	predicted=list()
	folds=loso(dataset,numberofsubjects)
	for fold in folds:
		train_set=list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy=list(row)
			test_set.append(row_copy)
			predicted=algorithm(train_set,test_set,5)
			results.append(predicted)
	return predicted

def euclideanDistance(instance1,instance2,length):
	distance = 0
	
	for x in range(length):
		a=(instance1[x])
		b=(instance2[x])
		distance+=pow((a-b),2)
		
	return math.sqrt(distance)
 

def algorithm(train_set,test_set,k1):
	distances=[]
	for i in range(len(test_set)):
		length=len(test_set[i])-1
		for k in range(len(train_set)):
			dist=euclideanDistance(test_set[i],train_set[k],length)
		
			distances.append((train_set[k],dist))
		distances.sort(key=operator.itemgetter(1))
		neighbors=[]
		for x in range(k1):
			neighbors.append(distances[x][0])
		
		return getResponse(neighbors)



def getResponse(neighbors):
	class1=0
	class0=0
	for x in range(len(neighbors)):
		response=neighbors[x][-1]
		if(response==1):
			class1=class1+1
		else:
			class0=class0+1
	print "Class 1=",class1
	print "Class 0=",class0
	if(class1>class0):
		selectedvote=1
	else:
		selectedvote=0
	print selectedvote
	return selectedvote


def main():
	filename="train_data2.csv"
	dataset=load_csv(filename)
	predicted=list()
	predicted=evaluate_algorithm(dataset,40)
	getaccuracy[]
	#print('Scores: %s' % scores)
	#print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

main()

