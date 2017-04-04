from scipy import stats
import pandas as pd
import numpy as np
import scipy
import copy
from sklearn.svm import SVC
import math
from sklearn import metrics

def splitdatatarget(dataframe):
    return dataframe.iloc[:,1:dataframe.shape[1]-1],dataframe.iloc[:,dataframe.shape[1]-1:dataframe.shape[1]]

def makefolds(X,Y,k):
    folds_y=list()
    folds_x=list()
    tempy=list()
    tempx=list()
    for i in range(1,X.shape[0]+1):
        tempx.append((X.iloc[i-1]).values.tolist())
        tempy.append((Y.iloc[i-1]).values.tolist())
        if i%k==0:
            folds_x.append(copy.deepcopy(tempx))
            folds_y.append(copy.deepcopy(tempy))
            tempy=list()
            tempx=list()
    return folds_x,folds_y

def mean(full_list_X,full_list_Y):
    rowx=list()
    rowy=list()
    meanX=list()
    meanY=list()
    for i in range(len(full_list_X)):
        x=full_list_X[i]
        #x1=
        for j in range(len(full_list_X[i][0])):
            y = zip(*x)[j]

            rowx.append(scipy.mean(y))
        temp = zip(*(full_list_Y[i]))
        y1 = list(temp[0])
        meanY.append(y1[0])
        meanX.append(rowx)
        rowx=list()
    return meanX,meanY

def trimmed_mean(full_list_X,full_list_Y):
    rowx=list()
    rowy=list()
    tmeanX=list()
    tmeanY=list()
    for i in range(len(full_list_X)):
        x=full_list_X[i]
        for j in range(len(full_list_X[i][0])):
            y = zip(*x)[j]
            rowx.append(stats.trim_mean(y,0.25))
        temp = zip(*(full_list_Y[i]))
        y1 = list(temp[0])
        tmeanY.append(y1[0])
        tmeanX.append(rowx)
        rowx=list()
    return tmeanX,tmeanY

def median(full_list_X,full_list_Y):
    rowx=list()
    rowy=list()
    medianX=list()
    medianY=list()
    for i in range(len(full_list_X)):
        x=full_list_X[i]
        #x1=
        for j in range(len(full_list_X[i][0])):
            y = zip(*x)[j]

            rowx.append(scipy.mean(y))
        temp = zip(*(full_list_Y[i]))
        y1 = list(temp[0])
        medianY.append(y1[0])
        medianX.append(rowx)
        rowx=list()
    return medianX,medianY

def interquartile_range(full_list_X,full_list_Y):
    rowx=list()
    rowy=list()
    iqrX=list()
    iqrY=list()
    for i in range(len(full_list_X)):
        x=full_list_X[i]
        rowx=(stats.iqr(x,axis=0)).tolist()
        temp = zip(*(full_list_Y[i]))
        y1 = list(temp[0])
        iqrY.append(y1[0])
        iqrX.append(rowx)
        rowx=list()
        rowy=list()

    return iqrX,iqrY

def standard_deviation(full_list_X,full_list_Y):
    rowx=list()
    rowy=list()
    stdX=list()
    stdY=list()
    for i in range(len(full_list_X)):
        x=full_list_X[i]
        rowx=(np.std(x,axis=0)).tolist()
        temp = zip(*(full_list_Y[i]))
        y1 = list(temp[0])
        stdY.append(y1[0])
        stdX.append(rowx)
        rowx=list()
        rowy=list()

    return stdX,stdY

def mad(data, axis=None):
    return np.mean(np.absolute(data-np.mean(data,axis)),axis)

def mean_absolute_deviation(full_list_X,full_list_Y):
    rowx=list()
    rowy=list()
    madX=list()
    madY=list()
    for i in range(len(full_list_X)):
        x=full_list_X[i]
        rowx=(mad(x,axis=0)).tolist()
        temp = zip(*(full_list_Y[i]))
        y1 = list(temp[0])
        madY.append(y1[0])
        madX.append(rowx)
        rowx=list()
        rowy=list()

    return madX,madY

def testtrainsplit(groups,i):
    test=list()
    train=list()
    for data in groups:
        for j in range(len(data)):
            if i==j:
                test.append(data[j])
            else:
                train.append(data[j])
    return train,test

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

def svm(groupsX,groupsY):
    predicted=list()
    expected=list()
    for i in range(len(groupsX[0])):
        X_train,X_test = testtrainsplit(groupsX,i)
        Y_train,Y_test = testtrainsplit(groupsY,i)
        model = SVC(kernel='linear')
        model.fit(X_train,Y_train)
        predicted.extend(model.predict(X_test))
        expected.extend(Y_test)
    accuracy_metric(expected,predicted)
    return predicted,expected


def main():
    k=26
    df=pd.read_csv('train_data2.csv',header=None)
    X, y = splitdatatarget(df)
    folds_x,folds_y = makefolds(X,y,k)
    datasetsX=list()
    datasetsY=list()
    for i in range(6):
        if i==0:
            datasetsX.append(mean(folds_x,folds_y)[0])
            datasetsY.append(mean(folds_x,folds_y)[1])
        if i==1:
            datasetsX.append(median(folds_x,folds_y)[0])
            datasetsY.append(median(folds_x,folds_y)[1])
        if i==2:
            datasetsX.append(trimmed_mean(folds_x, folds_y)[0])
            datasetsY.append(trimmed_mean(folds_x, folds_y)[1])
        if i==3:
            datasetsX.append(standard_deviation(folds_x, folds_y)[0])
            datasetsY.append(standard_deviation(folds_x, folds_y)[1])
        if i==4:
            datasetsX.append(mean_absolute_deviation(folds_x, folds_y)[0])
            datasetsY.append(mean_absolute_deviation(folds_x, folds_y)[1])
        if i==5:
            datasetsX.append(interquartile_range(folds_x, folds_y)[0])
            datasetsY.append(interquartile_range(folds_x, folds_y)[1])
    print datasetsY
    print len(datasetsX)

    for i in range(len(datasetsX)/2):
        groupsX = list()
        groupsY = list()
        print "s-loo",i+1,"-",i+4
        groupsX.append(datasetsX[i])
        groupsX.append(datasetsX[i+3])
        groupsY.append(datasetsY[i])
        groupsY.append(datasetsY[i+3])
        svm(groupsX,groupsY)


    print "s-loo(all)"
    svm(datasetsX,datasetsY)


if __name__ == '__main__':
    main()
