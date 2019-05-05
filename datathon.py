import pandas as pd
from sodapy import Socrata
from matplotlib import pyplot as plt
import os

#client = Socrata("controllerdata.lacity.org", None)
#results = client.get("969q-4gr3")
#results_df = pd.DataFrame.from_records(results)
#results_df.to_csv('datathon_all.csv')

results_df = pd.read_csv(u'City_Employee_Payroll.csv')

department_titles = results_df[u'Department Title'].unique()

num_of_departments = len(department_titles)

print (num_of_departments)

cols = results_df.columns

print (cols)

payroll_cols = cols[7:19]

figsize = 21,9
plt.figure(figsize=figsize)


average_basic_lifes = {}

for payroll_type in payroll_cols:
    os.mkdir(u'datathon/'+payroll_type+u'/')
    for i in range(0, num_of_departments, 4):
        for department_title in department_titles[i:i + 4]:
            average_basic_lifes.keys().append(department_title)
            average_basic_lifes[department_title] = results_df.loc[
                results_df[u'Department Title'] == department_title, payroll_type]

        average_basic_lifes_df = pd.DataFrame(average_basic_lifes)
        average_basic_lifes_df.boxplot()
        plt.savefig(u'datathon/'+payroll_type+u'/'+str(i))
        average_basic_lifes.clear()




#for department in average_basic_lifes.keys():
    #print (department+','+str(len(average_basic_lifes[department])))
    #plt.plot(range(len(average_basic_lifes[department])),average_basic_lifes[department], label = department, ls = ':')




