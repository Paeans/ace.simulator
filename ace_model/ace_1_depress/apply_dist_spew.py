#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

spew_dir = 'spew.data/counties/2010_ver1_45079'
data_prex = '2010_ver1_45079_synth_'
# pdata_file = spew_dir + '/' + data_prex + 'people.txt'
# hdata_file = spew_dir + '/' + data_prex + 'households.txt'
pdata_file = data_prex + 'people.txt'
hdata_file = data_prex + 'households.txt'

pdata = pd.read_csv(pdata_file, dtype = {'sp_id': object, 'sp_hh_id': object})
hdata = pd.read_csv(hdata_file, dtype = {'sp_id': object})

pdata = pdata.drop(columns=['serialno', 'stcotrbg', 'sp_school_id', 'sp_work_id'])
hdata = hdata[['sp_id', 'hh_income']]

phdata = pd.merge(pdata, hdata, left_on=['sp_hh_id'], right_on = ['sp_id'])

cdn_data = phdata[phdata['age'] <= 14]
# cdn_data['relate'].unique()
# cdn_data['race'].unique()
# cdn_data['sex'].unique()
# cdn_data['age'].unique()

income_cat = [0, 15000, 24999, 34999, 49999]

race_cat = [1,2,3,4,5]
ace_dist = {1:[0.2117052, 0.1637898, 0.1467698, 0.1533333, 0.1418293, 0.1085915], 
            2:[0.0834492, 0.0799530, 0.0733411, 0.0677146, 0.0846154, 0.0606601], 
            3:[0.1746032, 0.0763359, 0.1388889, 0.0909091, 0.1901408, 0.2162162], 
            4:[0.2500000, 0.1785714, 0.2033898, 0.1836735, 0.0958904, 0.1282051], 
            5:[0.2638889, 0.2857143, 0.3529412, 0.1785714, 0.2692308, 0.2266667]}

def cat_income(hh_income):
    if pd.isnull(hh_income):
        return 5
    if hh_income >= income_cat[0] and hh_income < income_cat[1]:
        return 0
    if hh_income >= income_cat[1] and hh_income < income_cat[2]:
        return 1
    if hh_income >= income_cat[2] and hh_income < income_cat[3]:
        return 2
    if hh_income >= income_cat[3] and hh_income < income_cat[4]:
        return 3
    if hh_income >= income_cat[4]:
        return 4

def cat_race(chd_race):
    if pd.isnull(chd_race):
        return 4
    if chd_race == 1:
        return 1
    if chd_race == 2:
        return 2
    if chd_race == 9:
        return 5
    return 4
    
def output_model(model):
    # agents = [a.output() 
    #           for i, sch_list in model.sch_group.items()
    #           for scheduler in sch_list 
    #           for a in scheduler.agents ]
    agents = [a.output() for a in model.schedule.agents]
    return pd.DataFrame(agents, columns=['id', 'age', 'race', 'income', 'ace'])
    # return agents

class Children(Agent):
    def __init__(self, chd_info, model, pos = None):
        unique_id = chd_info['sp_id_x']
        super().__init__(unique_id, model)
        self.pos = pos
        self.ace = 0
        self.age = chd_info['age']
        self.sex = chd_info['sex']
        self.income = chd_info['hh_income']
        self.race = chd_info['race']
        
    def step(self):
        p_aces = self.model.random.random()
        dist_cat = ace_dist[cat_race(self.race)][cat_income(self.income)]
        if p_aces < dist_cat:
            self.ace = 1
        else:
            self.ace = 0
            
    def get_cat(self):
        return cat_race(self.race), cat_income(self.income)
    
    def output(self):
        return [self.unique_id, self.age, self.race, self.income, self.ace]

class AceModel(Model):
    def __init__(self, chd_data):
        self.num_agents = len(chd_data)
        self.schedule = RandomActivation(self)
        self.sch_group = {x: [RandomActivation(self) 
                              for i in range(len(income_cat) + 1)] 
                          for x in race_cat}
        
        for i,chd_info in chd_data.iterrows():
            a = Children(chd_info, self)
            cat_r, cat_i = a.get_cat()
            self.schedule.add(a)
            self.sch_group[cat_r][cat_i].add(a)
            
        self.datacollector = DataCollector(
            model_reporters = {'Output': output_model}            
        )
        
    def step(self):
        for i, sch_list in self.sch_group.items():
            for scheduler in sch_list:
                self.reset_randomizer()
                scheduler.step()
        self.datacollector.collect(self)
        # self.schedule.step()

acemodel = AceModel(cdn_data)

for i in range(100):
    acemodel.step()
    res =acemodel.datacollector.model_vars['Output']
    res[-1].to_csv('simu_ace_'+str(i) + '.csv', index = False)



