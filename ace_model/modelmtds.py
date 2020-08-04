import brfss

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
#         ace = list(brfss.ace_list.values())[cdn.index]
        ace = random.choice(list(cdn.aces.keys()))
        p_aces = cdn.model.random.random()
        if p_aces < self.prop_ri[ace]:
            cdn.aces[ace] = 1
        else:
            cdn.aces[ace] = 0
#         cdn.index = (cdn.index + 1) % len(cdn.aces)

        
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
            print('index out of children size, resample')
            self.resampled = resample(self.samples, n_samples = len(cdn_group))
            self.index = 0
            
        for i in range(len(ace_keys)):
            cdn.aces[ace_keys[i]] = self.resampled[self.index, i]
        
        self.index += 1