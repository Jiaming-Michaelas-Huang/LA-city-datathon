import pandas as pd
from sodapy import Socrata
from matplotlib import pyplot as plt
import os
import numpy as ny

class department:
    def __init__(self,df,department_name):
        self.department_name = department_name
        self.data = df[u'Department Title' == department_name]
        self.years = df[u'Year'].unique()
        os.mkdir(u'datathon/'+department_name+u'/')
    def get_attr_year(self, attr_name,year):
        return self.data.loc[self.data[u'Year' == year],attr_name]
    def get_attr_years(self, attr_name, years):
        return self.data.loc[self.data[u'Year' in years], attr_name]
    def draw_boxplot(self, attr,save_or_not,year):
        results = {}
        for a in attr:
            results.keys().append(a)
            results[a] = self.data.loc[self.data[u'Year' in year],a]
        results_df = pd.DataFrame(results)
        results_df.boxplot()
        if save_or_not:
            plt.savefig(u'datathon/'+ self.department_name+u'/'+attr)
    def draw_histplot(self, attr,save_or_not,year):
        results = {}
        for a in attr:
            results.keys().append(a)
            results[a] = self.data.loc[self.data[u'Year' in year],a]
        results_df = pd.DataFrame(results)
        results_df.hist()
        if save_or_not:
            plt.savefig(u'datathon/'+ self.department_name+u'/'+attr)

    def mean(self, attr, year):
        results = {}
        means = {}
        for a in attr:
            results.keys().append(a)
            results[a] = self.data.loc[self.data[u'Year' in year], a]
            means.keys().append(a)
            means[a] = ny.mean(results[a])
        return means

    def variance(self, attr, year):
        results = {}
        vars = {}
        for a in attr:
            results.keys().append(a)
            results[a] = self.data.loc[self.data[u'Year' in year], a]
            vars.keys().append(a)
            vars[a] = ny.std(results[a])
        return vars





