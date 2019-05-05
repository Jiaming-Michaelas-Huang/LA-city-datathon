import pandas as pd
from sodapy import Socrata
from matplotlib import pyplot as plt
import os
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.linear_model import RANSACRegressor
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSCanonical
from sklearn.linear_model import HuberRegressor
import sklearn.preprocessing as preprocess
from sklearn.neural_network import MLPRegressor

def Run_Model(func,train_input, train_out_put, test_input, test_output):
    pre = func(train_input, train_out_put, test_input, test_output)
    return pre

def BP_Neural_Network(train_input, train_out_put, test_input, test_output):
    # parameter set
    input_level = train_input.columns.size
    output_level = train_out_put.culumns.size
    # construct BP Neural Network
    fnn = buildNetwork(input_level, 100, output_level, bias=True)

    # construct Training data

    ds = SupervisedDataSet(input_level, 2)

    input = train_input.as_matrix()
    output = train_out_put.as_matrix()

    for i in range(len(input)):
        print i
        ds.addSample(input[i], output[i])

    # check train set data

    print len(ds)

    print ds['input']
    print ds['target']

    # train model
    trainer = BackpropTrainer(fnn, ds)
    print trainer.trainEpochs(epochs=1)
    print fnn.params
    np.savetxt(u'network.txt', fnn.params)

    # input data
    input = test_input.as_matrix()
    print test_input

    # actual data
    output = test_output.as_matrix()
    print output

    pre = []
    for input1 in input:
        p = fnn.activate(input1)
        p1 = np.array(p)
        pre.append(p1)
    np.savetxt(u'prediction.txt', pre)
    pre = np.array(pre)
    # visualize prediction
    # adjust figure size
    plt.figure(figsize=(20, 8), dpi=80)
    # plot data
    x = range(len(output))
    plt.plot(x, output[:, 0], label=u'actual Q1', color='black')
    # plt.plot(x,output[:,1],label = u'actual Q2', color = 'blue')
    plt.plot(x, pre[:, 0], label=u'predict Q1', color='red')
    # plt.plot(x,pre[:,1],label = u'predict Q2',color='green')
    plt.legend()
    plt.savefig(u'result3')
    plt.show()
    return pre

def MLR(train_input, train_out_put, test_input, test_output):
    #set
    input = train_input
    output = train_out_put
    mlr1 = LinearRegression()
    mlr2 = LinearRegression()
    #train model
    mlr1.fit(input,output[:,0])
    mlr2.fit(input,output[:,1])
    #test model
    print mlr1
    print mlr2
    input = test_input
    pre1 = mlr1.predict(input)
    pre2 = mlr2.predict(input)
    # plot
    plt.figure()
    plt.scatter(x=pre1, y=test_output[:, 0])
    plt.show()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient

def DTRegression(train_input, train_out_put, test_input, test_output):
    #set
    dtr1 = DecisionTreeRegressor(max_depth=20)
    dtr2 = DecisionTreeRegressor(max_depth=20)
    input = train_input
    output = train_out_put
    # train model
    dtr1.fit(input,output[:,0])
    dtr2.fit(input,output[:,1])
    # predict
    input = test_input
    pre1 = dtr1.predict(input)*1.08
    pre2 = dtr2.predict(input)*1.08
    #plot
    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    #analyze
    num_of_efficeient=0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum]+pre2[rnum]
        actual_total = test_output[rnum,0]+test_output[rnum,1]
        if (np.abs(pre_total-actual_total)<=500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:'+str(num_of_efficeient/len(pre1))
    sse=[]
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i]+pre2[i] - test_output[i,0]-test_output[i,1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:'+str(sum_erro)
    return num_of_efficeient

def SVM(train_input, train_out_put, test_input, test_output):
    # set
    svm1 = svm.SVR()
    svm2 = svm.SVR()
    input = train_input
    output = train_out_put
    # train model
    svm1.fit(input, output[:,0])
    svm2.fit(input, output[:,1])
    # predict
    input = test_input
    pre1 = svm1.predict(input)
    pre2 = svm2.predict(input)
    # plot
    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient

def KNNRegression(train_input, train_out_put, test_input, test_output):
    # set
    knn1 = neighbors.KNeighborsRegressor()
    knn2 = neighbors.KNeighborsRegressor()
    input = train_input
    output = train_out_put
    # train model
    knn1.fit(input, output[:,0])
    knn2.fit(input, output[:,1])
    # predict
    input = test_input
    pre1 = knn1.predict(input)
    pre2 = knn2.predict(input)
    # plot
    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient

def RandomForestRegression(train_input, train_out_put, test_input, test_output):
    # set
    rf1 = ensemble.RandomForestRegressor(n_estimators=20)
    rf2 = ensemble.RandomForestRegressor(n_estimators=20)
    input = train_input
    output = train_out_put
    # train model
    rf1.fit(input, output[:,0])
    rf2.fit(input, output[:,1])
    # predict
    input = test_input
    pre1 = rf1.predict(input)
    pre2 = rf2.predict(input)
    # plot
    plt.figure()
    plt.scatter(x=pre1, y=test_output[:, 0])
    plt.show()
    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient

def Adaboost(train_input, train_out_put, test_input, test_output):
    # set
    ada1 = ensemble.AdaBoostRegressor(n_estimators=20)
    ada2 = ensemble.AdaBoostRegressor(n_estimators=20)
    input = train_input
    output = train_out_put
    # train model
    ada1.fit(input, output[:,0])
    ada2.fit(input, output[:,1])
    # predict
    input = test_input
    pre1 = ada1.predict(input)
    pre2 = ada2.predict(input)
    # plot
    # plot
    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient

def GRT(train_input, train_out_put, test_input, test_output):
    # set
    grt1 = ensemble.GradientBoostingRegressor(n_estimators=20)
    grt2 = ensemble.GradientBoostingRegressor(n_estimators=20)
    input = train_input
    output = train_out_put
    # train model
    grt1.fit(input, output[:,0])
    grt2.fit(input, output[:,1])
    # predict
    input = test_input
    pre1 = grt1.predict(input)
    pre2 = grt2.predict(input)
    # plot
    plt.figure()
    plt.scatter(x=pre1, y=test_output[:, 0])
    plt.show()
    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient

def Bagging(train_input, train_out_put, test_input, test_output):
    # set
    bag1 = MLPRegressor(solver="lbfgs",hidden_layer_sizes=(7,7,7),random_state=1)
    bag2 = MLPRegressor(solver="lbfgs",hidden_layer_sizes=(7,7,7),random_state=1)
    input = train_input
    output = train_out_put
    # train model
    bag1.fit(input, output[:,0])
    bag2.fit(input, output[:,1])
    # predict
    input = test_input
    pre1 = bag1.predict(input)
    pre2 = bag2.predict(input)
    # plot

    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient


def Robustness_Regression(train_input, train_out_put, test_input, test_output):
    ransac1 = RANSACRegressor(residual_threshold=400,max_trials=30000)
    ransac2 = RANSACRegressor(residual_threshold=400,max_trials=30000)
    ransac1.fit(train_input,train_out_put[:,0])
    ransac2.fit(train_input, train_out_put[:, 1])
    pre1 = ransac1.predict(test_input)
    pre2 = ransac2.predict(test_input)


    # plot
    plt.figure()
    plt.scatter(x=pre1, y=test_output[:, 0])
    plt.show()
    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient

def Ridge_Regression(train_input, train_out_put, test_input, test_output):
    r1 =linear_model.Ridge(alpha=15000000)
    r2 = linear_model.Ridge(alpha=15000000)
    r1.fit(train_input,train_out_put[:,0])
    r2.fit(train_input, train_out_put[:, 1])
    pre1 = r1.predict(test_input)
    pre2 = r2.predict(test_input)
    # plot
    plt.figure()
    plt.scatter(x=pre1, y=test_output[:, 0])
    plt.show()
    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient

def Kernal_Ridge_Regression(train_input, train_out_put, test_input, test_output):
    r1 =KernelRidge(kernel='rbf', alpha=0.1, gamma=10)
    r2 = KernelRidge(kernel='rbf', alpha=0.1, gamma=10)
    r1.fit(train_input,train_out_put[:,0])
    r2.fit(train_input, train_out_put[:, 1])
    pre1 = r1.predict(test_input)
    pre2 = r2.predict(test_input)
    # plot
    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient

def Lasso_Regression(train_input, train_out_put, test_input, test_output):
    r1 =linear_model.Lasso(alpha=15000000)
    r2 = linear_model.Lasso(alpha=15000000)
    r1.fit(train_input,train_out_put[:,0])
    r2.fit(train_input, train_out_put[:, 1])
    pre1 = r1.predict(test_input)
    pre2 = r2.predict(test_input)
    # plot
    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient





def ElasticNet_Regression(train_input, train_out_put, test_input, test_output):
    r1 =linear_model.Lars()
    r2 = linear_model.Lars()
    r1.fit(train_input,train_out_put[:,0])
    r2.fit(train_input, train_out_put[:, 1])
    pre1 = r1.predict(test_input)
    pre2 = r2.predict(test_input)
    # plot
    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient

def Bay_Regression(train_input, train_out_put, test_input, test_output):
    r1 =linear_model.BayesianRidge(alpha_1=10000,lambda_1=10000)
    r2 = linear_model.BayesianRidge()
    r1.fit(train_input,train_out_put[:,0])
    r2.fit(train_input, train_out_put[:, 1])
    pre1 = r1.predict(test_input)
    pre2 = r2.predict(test_input)
    # plot
    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient
def Logistic_Regression(train_input, train_out_put, test_input, test_output):
    r1 =linear_model.LogisticRegression()
    r2 = linear_model.LogisticRegression()
    r1.fit(train_input,train_out_put[:,0])
    r2.fit(train_input, train_out_put[:, 1])
    pre1 = r1.predict(test_input)
    pre2 = r2.predict(test_input)
    # plot
    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient

def PLS(train_input, train_out_put, test_input, test_output):
    r1 =PLSCanonical(algorithm='nipals', copy=True, max_iter=500, n_components=2,scale=True, tol=1e-06)
    r2 =PLSCanonical(algorithm='nipals', copy=True, max_iter=500, n_components=2,scale=True, tol=1e-06)
    r1.fit(train_input,train_out_put[:,0])
    r2.fit(train_input, train_out_put[:, 1])
    pre1 = r1.predict(test_input)
    pre2 = r2.predict(test_input)
    # plot
    plt.figure()
    plt.plot(np.cumsum(pre1), label="predict Q1")
    plt.plot(np.cumsum(pre2), label="predict Q2")
    plt.plot(np.cumsum(test_output[:, 0]), label="real Q1")
    plt.plot(np.cumsum(test_output[:, 1]), label="real Q2")
    plt.legend(loc="upper right")
    plt.show()
    # analyze
    num_of_efficeient = 0
    for rnum in range(len(pre1)):
        pre_total = pre1[rnum] + pre2[rnum]
        actual_total = test_output[rnum, 0] + test_output[rnum, 1]
        if (np.abs(pre_total - actual_total) <= 500):
            num_of_efficeient = num_of_efficeient + 1
    print u'the Matching ratio:' + str(num_of_efficeient / len(pre1))
    sse = []
    sum_mean = 0
    for i in range(len(pre1)):
        s = pre1[i] + pre2[i] - test_output[i, 0] - test_output[i, 1]
        s = np.square(s)
        sse.append(s)
        sum_mean += s
    sum_erro = np.sqrt(sum_mean / (len(pre1) - 1))
    print u'RMSE:' + str(sum_erro)
    return num_of_efficeient