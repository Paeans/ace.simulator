import brfss

import numpy as np
from sklearn.utils import resample

scbrfss = brfss.bfs_data('./brfss/SCBRFSS.csv')


class default_mtd():
    
    def __init__(self, race, income, cdn_group):
        self.race = race
        self.income = income
        self.prop_ri = {
            ace: scbrfss.get_prop(race, income, ace)[0] 
               for ace in brfss.ace_list.values()
        }
    
    def step_mtd(self, cdn):
        ace = random.choice(list(cdn.aces.keys()))
        p_aces = cdn.model.random.random()
        if p_aces < self.prop_ri[ace]:
            cdn.aces[ace] = 1
        else:
            cdn.aces[ace] = 0

        
class corr_rand_mtd(default_mtd):
    
    def __init__(self, race, income, cdn_group):
        super().__init__(race, income, cdn_group)
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


class bootstrap_mtd(default_mtd):
    
    def __init__(self, race, income, cdn_group):
        super().__init__(race, income, cdn_group)
        self.samples = scbrfss.get_value(race, income, 
                        list(brfss.ace_list.values())).to_numpy()[:, 2:]
        self.resampled = resample(self.samples, n_samples = len(cdn_group))
        self.index = 0
        
    def step_mtd(self, cdn):
        ace_keys = list(cdn.aces.keys())
        if self.index >= self.resampled.shape[0]:
            print('index large than children size, resample')
            self.resampled = resample(self.samples, n_samples = len(cdn_group))
            self.index = 0
            
        for i in range(len(ace_keys)):
            cdn.aces[ace_keys[i]] = self.resampled[self.index, i]
        
        self.index += 1
        
class bootstrap_mtd_nonan(bootstrap_mtd):
    
    def __init__(self, race, income, cdn_group):
        super().__init__(race, income, cdn_group)
        
        self.samples = self.samples[~np.isnan(self.samples).any(axis = 1)]
        self.resampled = resample(self.samples, n_samples = len(cdn_group))
        self.index = 0
        
class bootstrap_mtd_ptnan(bootstrap_mtd):
    
    def __init__(self, race, income, cdn_group):
        super().__init__(race, income, cdn_group)
        
        self.samples = self.samples[~np.isnan(self.samples).all(axis = 1)]
        self.resampled = resample(self.samples, n_samples = len(cdn_group))
        self.index = 0
        
        
class bootstrap_mtd_fill(default_mtd):
    
    def __init__(self, race, income, cdn_group, file_num = 1):
        super().__init__(race, income, cdn_group)
        
        self.samples = self.fill_nan(race, income, file_num)
        self.resampled = resample(self.samples, n_samples = len(cdn_group))
        self.index = 0
        
    def step_mtd(self, cdn):
        ace_keys = list(cdn.aces.keys())
        if self.index >= self.resampled.shape[0]:
            print('index large than children size, resample')
            self.resampled = resample(self.samples, n_samples = len(cdn_group))
            self.index = 0
            
        for i in range(len(ace_keys)):
            cdn.aces[ace_keys[i]] = self.resampled[self.index, i]
        
        self.index += 1
        
    def fill_nan(self, race, income, fill_num):
        ri_values = scbrfss.get_value(race, income, 
                        list(brfss.ace_list.values())).to_numpy()[:, 2:]        
#         ri_len = len(ri_values)
        
        all_one_values = ri_values[np.isnan(ri_values).sum(axis = 1) <= 1]
        all_len = 0
        
        for i in range(10):
            all_val = all_one_values[np.isnan(all_one_values).sum(axis = 1) == 0]
            if len(all_val) == all_len: # and all_len/ri_len > 0.5:
                break
                
#             if len(all_val) == all_len:
#                 fill_num -= 1
            
            all_len = len(all_val)
            for t in all_one_values:
                if ~np.isnan(t).any():
                    continue
                    
                similar_val = all_val[(all_val == t).sum(axis = 1) >= fill_num]
                if len(similar_val) == 0:
                    continue
                t[np.argwhere(np.isnan(t))[0][0]] = resample(similar_val[:,np.isnan(t)], n_samples = 1) #random.choice(similar_val[:,np.isnan(t)])
                
#         return all_one_values
        return all_one_values[np.isnan(all_one_values).sum(axis = 1) == 0]
        