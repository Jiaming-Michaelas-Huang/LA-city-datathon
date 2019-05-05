import pandas as pd
from sodapy import Socrata
from matplotlib import pyplot as plt
import numpy as np
import Models as model
import sklearn.preprocessing as preprocess
from sklearn.decomposition import PCA



if __name__ == '__main__':







    # read data
    department_df = pd.read_csv(u'data1/department_data.csv')
    job_df = pd.read_csv(u'data1/job_data.csv')
    individual_df = pd.read_csv(u'data1/dataset after preprocess.csv')
    individual_df1 = pd.read_csv(u'City_Employee_Payroll.csv')
    input_attributes = [u'pas_mean', u'pas_std', u'base_pay_mean', u'base_pay_std'
        , u'over_base_pay_mean', u'over_base_pay_std', u'fthcratio'
        , u'job_pas_mean',u'job_pas_mean_sd',u'job_base_mean',u'job_base_mean_sd',u'job_over_base_mean'
        , u'job_over_base_mean_sd',u'job_fthcratio'
        , u'Projected Annual Salary', u'Q1 Payments', u'Q2 Payments', u'Q3 Payments'
        , u'Q4 Payments', u'Base Pay']
    model_names = [u'BP_Neural_Network', u'MLR', u'DTRegression', u'SVM', u'KNNRegression', u'RandomForestRegression',
                   u'Adaboost', u'GRT', u'Bagging',u'Robustness_Regression',u'Ridge_Regression'
                ,u'Kernal_Ridge_Regression',u'Lasso_Regression',u'ElasticNet_Regression'
                ,u'Bay_Regression',u'Logistic_Regression',u'PLS']
    models_pre = {}
    for model_name in model_names:
        models_pre.keys().append(model_name)
        models_pre[model_name] = []
    real_target = []
    # rolling based training and testing
    train_years = [2013]
    test_years = [2014]
    models=[model_names[9]]

    for i in range(len(test_years)):
        test_year = test_years[i]
        # train input
        department_df_train = department_df.loc[department_df[u'Year'].isin(train_years)]
        job_df_train = job_df.loc[job_df[u'Year'] .isin(train_years)]
        individual_df_train = individual_df.loc[individual_df[u'Year'].isin(train_years)]
        t = train_years[len(train_years)-1]+1
        df_train_target = individual_df.loc[(individual_df[u'Year'] == t), [u'Record Number', u'Q1 Payments', u'Q2 Payments']]
        df_train_target = df_train_target.rename(columns={u'Q1 Payments':u'Pre Q1 Payments',u'Q2 Payments':u'Pre Q2 Payments'})

        # merge dataset
        df = pd.merge(right=department_df_train, left=individual_df_train, how='left',left_on='Payroll Department',right_on='Payroll.Department')
        df = pd.merge(right=job_df_train, left=df, how='left',left_on='Job Class',right_on='Job.Class')
        df = pd.merge(df,df_train_target,on=u'Record Number')
        #df = pd.merge(df, job_df_train, on='Job Class')
        #df.to_csv('data1/merged.csv')

        # build input data
        input_train = np.array(df[input_attributes])
        target_train = np.array(df[[u'Pre Q1 Payments',u'Pre Q2 Payments']])

        #test input
        department_df_test = department_df.loc[department_df[u'Year'] == test_year]
        job_df_test = job_df.loc[job_df[u'Year'] == test_year]
        individual_df_test1 = individual_df1.loc[individual_df1[u'Year'] == test_year]
        t = test_year + 1
        record_number = np.array(individual_df1[u'Record Number'].unique())
        df_test_target = individual_df1.loc[
            (individual_df1[u'Year'] == t), [u'Record Number', u'Q1 Payments', u'Q2 Payments']]
        df_test_target = df_test_target.rename(
            columns={u'Q1 Payments': u'Pre Q1 Payments', u'Q2 Payments': u'Pre Q2 Payments'})
        print df_test_target.loc[df_test_target[u'Record Number']==u'1060000000']


        # merge dataset
        df = pd.merge(right=department_df_test, left=individual_df_test1, how='left', left_on='Payroll Department',
                      right_on='Payroll.Department')
        df = pd.merge(right=job_df_test, left=df, how='left', left_on='Job Class', right_on='Job.Class')
        print df[u'Record Number']
        print df_test_target[u'Record Number']
        df = pd.merge(left=df, right=df_test_target,how='right', on=u'Record Number')
        df = df.fillna(0)
        # df = pd.merge(df, job_df_train, on='Job Class')
        #df.to_csv('data1/newdata.csv')
        # build input data
        input_test = np.array(df[input_attributes])
        target_test = np.array(df[[u'Pre Q1 Payments',u'Pre Q2 Payments']])


        #real_target = real_target + target_test
        pca = PCA(n_components=6)
        pca.fit(input_train)
        input_train = pca.transform(input_train)
        input_test = pca.transform(input_test)

        Pre = model.Run_Model(model.Robustness_Regression,input_train,target_train,input_test,target_test)
        df[u'Q1+Q2'] = Pre
        df2 = df[[u'Record Number',u'Q1+Q2']]
        df2.to_csv(u'data1/re2.csv')
