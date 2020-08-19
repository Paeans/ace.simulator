
import pandas as pd
import numpy as np

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from collections import OrderedDict

import brfss

class Children(Agent):
    
    def __init__(self, chd_info, model, 
                 step_method, pos = None):
        unique_id = chd_info['sp_id_x']
        super().__init__(unique_id, model)
        self.mtd = step_method
        self.pos = pos
        self.age = chd_info['age']
        self.sex = chd_info['sex']
        self.income = chd_info['_INCOMG']
        self.race = chd_info['_RACE_G1']        
        self.aces = {ace:0 for ace in brfss.ace_list.values()}
        self.index = 0
        
    def step(self):
        self.mtd.step_mtd(self)
            
    def get_cat(self):
        return self.race, self.income
    
    def output(self):        
        return [self.unique_id, self.age, self.race, self.income, ] + \
                list(self.aces.values())

class HouseHold(Agent):
    
    def __init__(self, house_id, income_code, model, step_method, trans_rate = 0.2, pos = None):
        unique_id = house_id
        super().__init__(unique_id, model)
        self.trans_rate = trans_rate
        self.mtd = step_method
        self.pos = pos
        self.income = income_code
        self.persons = OrderedDict()        
        
        self.steps = 0
        
    def add_person(self, chd_info):
        person = Children(chd_info, self.model, self.mtd)
        self.persons[person.unique_id] = person
        pass
    
    def get_persons(self):
        return self.persons.values()
    
    def get_adults(self):
        res = []
        for p in self.persons.values():
            if p.age >= 18:
                res.append(p)
                
        return res
    
    def get_childs(self):
        res = []
        for p in self.persons.values():
            if p.age < 18:
                res.append(p)
                
        return res
    
    def step(self):
        # print(self.unique_id)
        if self.steps == 0: # init adults
            [p.step() for p in self.get_adults()]                
            self.steps += 1
            return
        for c in self.get_childs():
            for p in self.get_adults():
                for ace in p.aces.keys():
                    if p.aces[ace] == 0 or np.isnan(p.aces[ace]):
                        continue
                    p_trans = self.model.random.random()
                    if p_trans <= self.trans_rate:
                        c.aces[ace] = p.aces[ace]
        self.steps += 1
        
class HouseModel(Model):
    def __init__(self, chd_data, race, income, step_method):

        self.schedule = RandomActivation(self)
        self.step_method = step_method(race, income, chd_data)
        
        if not race == 0:
            chd_data = chd_data[(chd_data['_RACE_G1']) == race]
            
        if not income == 0:
            chd_data = chd_data[(chd_data['_INCOMG']) == income]
            
        self.num_agents = len(chd_data)    
        for i,chd_info in chd_data.iterrows():
            house_id = chd_info['sp_hh_id']
            income_code = chd_info['_INCOMG']
            if not house_id in self.schedule._agents.keys():
                self.schedule.add(
                    HouseHold(house_id, income_code, self, self.step_method)
                )
            self.schedule._agents[house_id].add_person(chd_info)
            
        self.datacollector = DataCollector(
            model_reporters = {'Output': 
               lambda x: pd.DataFrame(
                   [p.output() for a in x.schedule.agents for p in a.get_persons()], 
                   columns=['id', 'age', '_RACE_G1', '_INCOMG'] + 
                   list(brfss.ace_list.values())
               )}
        )
        
    def step(self):
        self.reset_randomizer()
        self.schedule.step()
        self.datacollector.collect(self)        
        
    
class AceModel(Model):
    def __init__(self, chd_data, 
                 race, income, step_method):

        self.schedule = RandomActivation(self)
        self.step_method = step_method(race, income, chd_data)
        
        if not race == 0:
            chd_data = chd_data[(chd_data['_RACE_G1']) == race]
            
        if not income == 0:
            chd_data = chd_data[(chd_data['_INCOMG']) == income]
            
        self.num_agents = len(chd_data)    
        for i,chd_info in chd_data.iterrows():            
            self.schedule.add(
                Children(chd_info, self, self.step_method)
            )
            
        self.datacollector = DataCollector(
            model_reporters = {'Output': 
               lambda x: pd.DataFrame(
                   [a.output() for a in x.schedule.agents], 
                   columns=['id', 'age', '_RACE_G1', '_INCOMG'] + 
                   list(brfss.ace_list.values())
               )}
        )
        
    def step(self):
        self.reset_randomizer()
        self.schedule.step()
        self.datacollector.collect(self)


def simulate_stack(mtd, cdl_data, model = AceModel, step_num = 1000, display = False):

    acemodel_list = [
        model(cdl_data[
            (cdl_data[['_RACE_G1', '_INCOMG']] == [r,i]).all(axis = 1)
        ], r, i, step_method = mtd)
        for r in list(brfss.race_list.keys())[1:]
        for i in list(brfss.income_list.keys())[1:]
                    ]
    
    for i in range(step_num):
        [acemodel.step() for acemodel in acemodel_list]

    result = [res for res in [
            acemodel.datacollector.model_vars['Output'][-1]
            for acemodel in acemodel_list]
                if len(res) > 0
            ]
    
    result_df = pd.concat(result, axis = 0, 
                          ignore_index = True)
    
    if not display:
        return result_df
    
    result_bfs = bfs_data(result_df)
    
    for r in list(brfss.race_list.keys())[1:]:
        for i in list(brfss.income_list.keys())[1:]:
            print('RACE: ' + str(brfss.race_list[r]) + ' INCOME: ' + str(brfss.income_list[i]))
            display(result_bfs.get_corr_mat(r,i))
    return result_df