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
		dataset.pop(0)
		total=len(dataset)
		print total
		for i in range(0,total):
			for value in dataset[i]:
				value=float(value)
		for i in range(0,total):
			dataset[i].pop()
		return dataset

def cross_validation_split(dataset,n_folds):
	dataset_split=list()
	dataset_copy=list(dataset)
	fold_size=int(len(dataset)/n_folds)
	for i in range(n_folds):
		fold=list()
		while len(fold)<fold_size:
			index=randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split


def accuracy_metric(test_set, predicted):
	correct = 0
	for i in range(len(test_set)):
		if test_set[i][-1] == predicted[i]:
			correct += 1
	return correct / float(len(test_set)) * 100.0
 

def evaluate_algorithm(dataset,algorithm,n_folds):
	folds=cross_validation_split(dataset,n_folds)
	for fold in folds:
		train_set=list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy=list(row)
			test_set.append(row_copy)
		predicted=algorithm(train_set,test_set,10)
		accuracy = accuracy_metric(test_set, predicted)
		scores.append(accuracy)
		return predicted




def euclideanDistance(instance1,instance2,length):
	distance = 0
	for x in range(length):
		#print x
		instance1[x]=float(instance1[x])
		instance2[x]=float(instance2[x])
		distance+=pow((instance1[x]-instance2[x]),2)
		
	return math.sqrt(distance)
 

def algorithm(dataset,test_set,k1):
	distances=[]
	for i in range(len(test_set)):
	i=0
	length=len(test_set[i])-1
	print test_set[0]
	for k in range(len(train_set)):
		train_set[k]=train_set[k].pop()
		print train_set[k]
		dist=euclideanDistance(test_set[i],train_set[k],length)
		print dist
		distances.append((train_set[k],dist))
	print distances
	distances.sort(key=operator.itemgetter(1))
	print distances
	neighbors=[]
	for x in range(k1):
		neighbors.append(distances[x][0])
	print neighbors
	return getResponse(neighbors)


def getResponse(neighbors):
	class1=0
	class0=0
	for x in range(len(neighbors)):
		response=neighbors[x]
		if(response==1):
			class1=class1+1
		else:
			class0=class0+1
	if(class1>class0):
		selectedvote=1
	else:
		selectedvote=0
	return  selectedvote




def main():
	filename="train_data2.csv"
	dataset=load_csv(filename)
	print dataset
	k=10
	algorithm(dataset,10)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

main()
