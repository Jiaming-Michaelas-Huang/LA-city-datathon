#data preprocess

#load package

import pandas as pd
from sodapy import Socrata
from matplotlib import pyplot as plt
import numpy as np

# load row data
results_df = pd.read_csv(u'City_Employee_Payroll.csv')



# default handling
#fillna payroll department
results_df[u'Payroll Department'] = results_df[u'Payroll Department'].fillna(69)

#fill projected annual salary with hourly or event rate
results_df[u'work hour'] = results_df[u'Projected Annual Salary']/results_df[u'Hourly or Event Rate']
#results_df[u'work hour'] = results_df[u'work hour'].fillna(2088)
#results_df.loc[(results_df[u'work hour']>3000)|(results_df[u'work hour']<2000), u'work hour'] = 2088
results_df[u'work hour'] = results_df[u'work hour'].fillna(-1)
results_df[u'Hourly or Event Rate']= results_df[u'Hourly or Event Rate'].fillna(-1)
results_df.loc[(results_df[u'work hour']<100)&(results_df[u'Hourly or Event Rate']==-1),u'Hourly or Event Rate'] = results_df.loc[(results_df[u'work hour']<100)&(results_df[u'Hourly or Event Rate']==-1),u'Projected Annual Salary']/2088
results_df.loc[(results_df[u'work hour']<100)&(results_df[u'Hourly or Event Rate']!=-1),u'Projected Annual Salary'] = results_df.loc[(results_df[u'work hour']<100)&(results_df[u'Hourly or Event Rate']==-1),u'Hourly or Event Rate']*2088

#overbase pay
results_df[u'Overbase Payments'] = results_df[u'Total Payments']-results_df[u'Base Pay']

#overbase pay ratio
results_df[u'% Over Base Pay'] = results_df[u'Overbase Payments']/results_df[u'Base Pay']



# exceptional handling

#select normal data
results_df = results_df[results_df[u'Projected Annual Salary'].notna()]
results_df = results_df[((results_df[u'Q1 Payments'])>(results_df[u'Projected Annual Salary'])/6)
                        &((results_df[u'Q1 Payments'])<(results_df[u'Projected Annual Salary'])/2)
                        &((results_df[u'Q2 Payments'])>(results_df[u'Projected Annual Salary'])/6)
                        &((results_df[u'Q2 Payments'])<(results_df[u'Projected Annual Salary'])/2)
                        &((results_df[u'Q3 Payments'])>(results_df[u'Projected Annual Salary'])/6)
                        &((results_df[u'Q3 Payments'])<(results_df[u'Projected Annual Salary'])/2)
                        &((results_df[u'Q4 Payments'])>(results_df[u'Projected Annual Salary'])/6)
                        &((results_df[u'Q4 Payments'])<(results_df[u'Projected Annual Salary'])/2)]


#output dataset
results_df.to_csv(u'data1/dataset after preprocess.csv')
print len(np.array(results_df))



