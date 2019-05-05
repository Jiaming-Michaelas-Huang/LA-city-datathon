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
import  multiprocessing
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier




def integral(data):
    if data == -1:
        return -1
    else:
        if (ord(data) - 48) > 16:
            return ord(data) - 55
        else:
            return ord(data) - 48

def addsample(data,ds):
    x = 0

def grade_level(row):
    df_need = results_df[results_df[u'Year'] == row[u'Year']]
    df_need = df_need[df_need[u'Job Class'] == row[u'Job Class']]
    row[u'Grade Level'] = df_need[row[u'Pay Grade'] <= df_need[u'Pay Grade']].count() / df_need[u'Pay Grade'].count()
    print row[u'Grade Level']
    return row

def tmp_func(df):
 df = df.apply(addsample,axis = 1)
 return df

def apply_parallel(df_grouped, func):
 results = Parallel(n_jobs=-1)(delayed(func)(group) for name, group in df_grouped)
 return pd.concat(results)

if __name__ == '__main__':

    #read raw data
    results_df = pd.read_csv(u'City_Employee_Payroll.csv')

    # pay grade fillna

    # pay grade adjustment
    # pay grade standarize
    results_df[u'Pay Grade'] = results_df[u'Pay Grade'].fillna(-1)
    results_df[u'Pay Grade'] = results_df[u'Pay Grade'].apply(integral)
    grade_df = results_df[[u'Row ID',u'Pay Grade']]
    #results_df.apply(grade_level,axis=1)

    #payroll department adjustment
    results_df[u'Payroll Department'] = results_df[u'Payroll Department'].fillna(69)
    num_department = len(results_df[u'Payroll Department'].unique())
    #dummy_variable
    payroll_department_df = pd.get_dummies(results_df[[u'Row ID',u'Payroll Department']], prefix=u'Department :',columns=[u'Payroll Department'])

    #employment type adjustment
    #dummy variable
    num_employment_type = len(results_df[u'Employment Type'].unique())
    employment_type_df = pd.get_dummies(results_df[[u'Row ID',u'Employment Type']], prefix=u'Employment Type', columns=[u'Employment Type'])
    #job class adjustment
    l = len(results_df[u'Job Class'])
    num_job_class = len(results_df[u'Job Class'].unique())
    #dummy variable
    job_class_df = pd.get_dummies(results_df[[u'Row ID',u'Job Class']], prefix=u'Job Class',columns=[u'Job Class'])






    #results_df.to_csv(u'modified_train_data.csv')

    df1 = pd.merge(job_class_df,employment_type_df,on=u'Row ID')
    df1 = pd.merge(df1,payroll_department_df,on=u'Row ID')
    df1 = pd.merge(df1, grade_df)
    df_final = pd.merge(df1,results_df[[u'Row ID',u'Year',u'Q1 Payments', u'Q2 Payments']])
    train_df = df_final[df_final[u'Year'].isin([u'2013', u'2014', u'2015', u'2016'])]
    train_df.drop([u'Row ID',u'Year'], axis=1, inplace=True)

    #output data
    output_df = train_df[[u'Q1 Payments',u'Q2 Payments']]
    output = output_df.as_matrix()
    print output

    #input data
    input_df = train_df.drop([u'Q1 Payments',u'Q2 Payments'],axis=1, inplace=False)
    input = input_df.as_matrix()
    print input

    # construct BP Neural Network
    fnn = buildNetwork(num_department+num_employment_type+num_job_class+1, 100, 2,bias = True)

    # construct Training data

    ds = SupervisedDataSet(num_department+num_employment_type+num_job_class+1, 2)

    for i in range(len(input)):
        print i
        ds.addSample(input[i],output[i])

    '''''
    for rownum in range(train_df.iloc[:, 0].size):
        data = train_df.iloc[rownum, :]
        if data[u'Q1 Payments'] > 0 and data[u'Q2 Payments'] > 0:
            print rownum
            ds.addSample(data.drop([u'Q1 Payments',u'Q2 Payments']),
                         (data[u'Q1 Payments'], data[u'Q2 Payments']))
                         
    '''''

    # check train set data

    print len(ds)

    print ds['input']
    print ds['target']

    # train model
    trainer = BackpropTrainer(fnn, ds)
    print trainer.trainEpochs(epochs=1)
    print fnn.params
    np.savetxt(u'network.txt',fnn.params)

    #test module

    test_df = df_final[df_final[u'Year'].isin([u'2017'])]
    test_df.drop([u'Row ID', u'Year'], axis=1, inplace=True)

    # input data
    input_df = train_df.drop([u'Q1 Payments', u'Q2 Payments'], axis=1, inplace=False)
    input = input_df.as_matrix()
    print input

    # actual data
    output_df = train_df[[u'Q1 Payments', u'Q2 Payments']]
    output = output_df.as_matrix()
    print output

    pre = []
    for input1 in input:
        p = fnn.activate(input1)
        p1 = np.array(p)
        pre.append(p1)
    np.savetxt(u'prediction.txt',pre)
    pre = np.array(pre)
    #visualize prediction
    #adjust figure size
    plt.figure(figsize=(20, 8), dpi=80)
    #plot data
    x= range(len(output))
    plt.plot(x,output[:,0],label = u'actual Q1',color = 'black')
    #plt.plot(x,output[:,1],label = u'actual Q2', color = 'blue')
    plt.plot(x,pre[:,0],label = u'predict Q1', color = 'red')
    #plt.plot(x,pre[:,1],label = u'predict Q2',color='green')
    plt.legend()
    plt.savefig(u'result3')
    plt.show()

