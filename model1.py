import pandas as pd
from sodapy import Socrata
from matplotlib import pyplot as plt
import os
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import  SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import numpy as np


results_df = pd.read_csv(u'City_Employee_Payroll.csv')

results_df[u'Payroll Department']=results_df[u'Payroll Department'].fillna(69)


# construct BP Neural Network
fnn = buildNetwork(3,5,2)

# construct Training data

ds = SupervisedDataSet(3,2)

train_df = results_df.loc[u'Year' in (u'2013',u'2014',u'2015',u'2016')]

for rownum in range(train_df.iloc[:,0].size):
    data = train_df.iloc[rownum,:]
    if data[u'Q1 Payments'] > 0 & data[u'Q2 Payments']>0:
        ds.addSample((data[u'Department Title Code'],data[u'Employment Type Code'],data[u'Pay Grade']),(data[u'Q1 Payments'],data[u'Q2 Payments']))
#check train set data

print len(ds)

print ds['input']
print ds['target']

# train model

trainer = BackpropTrainer(fnn,ds)






