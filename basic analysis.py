import pandas as pd
from sodapy import Socrata
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

results_df = pd.read_csv(u'data1/dataset after preprocess.csv')

df_y = results_df.loc[results_df[u'Year']==2013,[u'Record Number',u'Payroll Department',u'Job Class',u'Projected Annual Salary']]
#deterministic or dynamic
for y in [2014,2015,2016,2017]:
    df_y1 = results_df.loc[results_df[u'Year']==y,[u'Record Number',u'Payroll Department',u'Job Class',u'Projected Annual Salary']]
    df_y = pd.merge(df_y,df_y1,on=u'Record Number')


df_y.to_csv(u'deter.csv')



#results_df = results_df.loc[results_df[u'Employment Type']==u'Full Time']
results_df = results_df.loc[(results_df[u'Q1 Payments']>0) & (results_df[u'Q3 Payments']>0)]
r_num = len(np.array(results_df[u'Projected Annual Salary']))
plt.plot(results_df['Q1 Payments'],results_df['Q3 Payments'])
plt.show()

results_df = results_df.loc[results_df[u'Year']==2016]
results_df[u'Payroll Department'] = results_df[u'Payroll Department'].fillna(69)
rrr = np.array(results_df[u'Projected Annual Salary'])
for i in range(len(rrr)):
    rrr[i] = (rrr[i]-rrr.mean())/rrr.std()

plt.hist(rrr,bins=500)
plt.show()


#discover relationship between job class, department and projected anuual salary
job_class = np.array(results_df[u'Job Class'].unique())
department_class = np.array(results_df[u'Payroll Department'].unique())
job_class_department = []
for department in department_class:
    job_class_for_department = np.array(results_df.loc[results_df[u'Payroll Department']==department, u'Job Class'].unique())
    for job in job_class_for_department:
        job_class_department.append([department,job])
for job_department in job_class_department:
    a = results_df.loc[results_df[u'Job Class']==job_department[1]]
    b = a.loc[a[u'Payroll Department'] == job_department[0]]
    #projected annual salary for different pay grade
    pay_grade = sorted(np.array(b[u'Pay Grade'].unique()))
    #plot different pay grade
    plt.figure(figsize=(20,8))
    for grade in pay_grade:
        projected_annual = np.array(b.loc[b[u'Pay Grade']==grade, u'Projected Annual Salary']).mean()
        plt.plot([0,1],[projected_annual,projected_annual], label=str(grade))
    plt.legend()
    plt.show()
    #plt.savefig('2013/'+str(job_department[0])+u':'+str(job_department[1])+u'.jpg')
    plt.clf()




#same job different department
#budget for different department
colors = {2013:'red',2014:'yellow',2015:'blue',2016:'green',2017:'purple'}
budget = {}
plt.figure(figsize=(50, 8))
department_class = np.array(results_df[u'Payroll Department'].unique())
width=10
for department in department_class:
    budget.keys().append(department)
    budget[department] = []
for year in [2013,2014,2015,2016,2017]:
    results_df1 = results_df.loc[results_df[u'Year'] == year]
    for department in department_class:
        budg = np.array(
            results_df1.loc[results_df1[u'Payroll Department'] == department, u'Total Payments']).sum() / len(
            np.array(results_df1.loc[results_df1[u'Payroll Department'] == department]))
        if np.isnan(budg):
            budget[department] = 0
        else:
            budget[department] = budg
    x = range(1,100*len(department_class),100)
    plt.bar(x,budget)








#discover different job in department
department_class = np.array(results_df[u'Payroll Department'].unique())
for department in department_class:
    df = results_df.loc[results_df[u'Payroll Department'] == department, u'Projected Annual Salary']
    plt.bar()
    plt.show()
    plt.clf()




#discover job class and base_payment
job_class = np.array(results_df[u'Job Class'].unique())
results_df2 = results_df.loc[(results_df[u'Q1 Payments']!=0) & (results_df[u'Q2 Payments']!=0) & (results_df[u'Q3 Payments']!=0) & (results_df[u'Q4 Payments']!=0)]
for job in job_class:
    df2 = results_df.loc[results_df[u'Job Class']==job, u'Base Pay']
    plt.plot(df2,label = job)
plt.show()


#discover relationship betwwen projected annual value and Q1 Payments etc

df1 = results_df[[u'Projected Annual Salary',u'Q1 Payments']]
df1 = df1.dropna()
df1=df1.loc[df1[u'Q1 Payments']!=0]
dataset_x = np.array(df1[u'Projected Annual Salary']).reshape(-1,1)
dataset_y = np.array(df1[u'Q1 Payments']).reshape(-1,1)
reg = LinearRegression().fit(dataset_x,dataset_y)
print reg.score(dataset_x,dataset_y)
plt.scatter(dataset_x, dataset_y,  color='black')
plt.plot(dataset_x, reg.predict(dataset_x), color='red', linewidth=1)
plt.show()












