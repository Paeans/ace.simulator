
import pandas as pd
import random

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import brfss

scbrfss = brfss.bfs_data('./brfss/SCBRFSS.csv')


class default_mtd():
    
    def __init__(self, race, income):
        self.race = race
        self.income = income
        self.prop_ri = {
            ace: scbrfss.get_prop(race, income, ace)[0] 
               for ace in brfss.ace_list.values()
        }
    
    def step_mtd(self, cdn):
#         ace = list(brfss.ace_list.values())[cdn.index]
        ace = random.choice(list(cdn.aces.keys()))
        p_aces = cdn.model.random.random()
        if p_aces < self.prop_ri[ace]:
            cdn.aces[ace] = 1
        else:
            cdn.aces[ace] = 0
#         cdn.index = (cdn.index + 1) % len(cdn.aces)

        
class corr_rand_mtd(default_mtd):
    
    def __init__(self, race, income):
        super().__init__(race, income)
        self.corr_mat = scbrfss.get_corr_mat(race, income)
        self.corr_index = {self.corr_mat.index[x]:x 
                           for x in 
                          range(len(self.corr_mat.index))}
        self.corr_columns = {self.corr_mat.columns[x]:x 
                           for x in 
                          range(len(self.corr_mat.columns))}
        self.corr_mat = self.corr_mat.to_numpy()
        
    def step_mtd(self, cdn):
        
        ace = random.choice(list(cdn.aces.keys()))
        
#         rel_aces = [k for k,v in cdn.aces.items() if v == 1]
        rel_aces = cdn.aces.items()
        p_aces = cdn.model.random.random()
        if p_aces < self.cal_prop(ace, rel_aces):
            cdn.aces[ace] = 1
        else:
            cdn.aces[ace] = 0
            
    def cal_prop(self, ace, rel_aces = None):
        if rel_aces == None or len(rel_aces) == 0:
            return self.prop_ri[ace]
        corrs = 0
        index = self.corr_index[ace]
        for k,v in rel_aces:
            if k == ace:
                break
            cr = self.corr_mat[index, self.corr_columns[k]]
            if v == 0:
                corrs -= cr
            else:
                corrs += cr
#         return self.prop_ri[ace] * (1 + corrs/len(rel_aces))
        return self.prop_ri[ace]  + corrs/len(rel_aces)
            
        

class Children(Agent):
    
    def __init__(self, chd_info, model, 
                 step_method, pos = None):
        unique_id = chd_info['sp_id_x']
        super().__init__(unique_id, model)
        self.mtd = step_method
        self.pos = pos
        # self.ace = 0
        self.age = chd_info['age']
        self.sex = chd_info['sex']
        self.income = chd_info['_INCOMG']
        self.race = chd_info['_RACE_G1']        
        self.aces = {ace:0 for ace in brfss.ace_list.values()}
        self.index = 0
#         self.prop_ri = prop_ri
        
    def step(self):
        self.mtd.step_mtd(self)
            
    def get_cat(self):
        return self.race, self.income
    
    def output(self):        
        return [self.unique_id, self.age, self.race, self.income, ] + \
                list(self.aces.values())
    
class AceModel(Model):
    def __init__(self, chd_data, 
                 race, income, step_method):
        self.num_agents = len(chd_data)
        self.schedule = RandomActivation(self)
        self.step_method = step_method(race, income)
        
        if not race == 0:
            chd_data = chd_data[(chd_data['_RACE_G1']) == race]
            
        if not income == 0:
            chd_data = chd_data[(chd_data['_INCOMG']) == income]
            
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
        