
import pandas as pd
import random

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import brfss

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
        self.step_method = step_method(race, income, chd_data)
        
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


def simulate_stack(mtd, cdl_data, step_num = 1000, display = False):

    acemodel_list = [
        AceModel(cdl_data[
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