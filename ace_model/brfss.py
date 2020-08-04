
import math
import numpy as np
import pandas as pd

from scipy.stats import norm

import matplotlib.pyplot as plt

ace_list = {1:'depress', 2:'alcoabuse', 3:'drugabuse', 4:'prison',
            5:'patdivorce', 6:'phyabuse1', 7:'phyabuse2', 8:'verbalabuse',
            9:'sexabuse1', 10:'sexabuse2', 11:'sexabuse3', 12:'foodinsecure'} 

groupa = list(ace_list.values())[0:5]
groupb = list(ace_list.values())[5:-1]
groupc = list(ace_list.values())[-1]

race_list = {0:'All', 1:'White', 2:'Black', 3:'Hispanic', 4:'Other', 5:'Multi'}

income_list = {0:'All', 1:'< 15000', 2:'15000 - 24999', 3:'25000 - 34999', 
               4:'35000 - 49999', 5:'50000 +', 9:'Don\'t Know'}

# generate combinations of aces
def comb(aces, n):
    res = []
    if n <= 0:
        return [[]]
    if n > len(aces):
        return comb(aces, len(aces))
    for i in range(len(aces) - n + 1):
        res += [[aces[i]] + x for x in comb(aces[i+1:], n - 1)]
    return res

# cast the aces code in brfss, ori_code to 0 -> No, 1 -> Yes
def cat_code(ori_code, *args):    
    
    if pd.isna(ori_code) or len(args) == 0:
        return ori_code
    
    col_name = args[0]
    if col_name not in list(ace_list.values()):
        return ori_code
    
    if (col_name in groupa and ori_code == 2) or \
        (col_name in groupb and ori_code == 1) or \
        (col_name in groupc and ori_code == 1):
        return 0
    if (col_name in groupa and ori_code == 1) or \
        (col_name in groupb and ori_code in [2,3]) or \
        (col_name in groupc and ori_code in [2,3,4,5]):
        return 1    
    return np.NaN

def cal_prop(df, *aces):
    if not aces:
        return np.NaN, np.NaN
    aces_values = df[list(aces)]
    k = (aces_values == 1).all(axis = 1).sum()
    n = (aces_values.isin([0,1])).all(axis = 1).sum()    

    if n == 0:
        return np.NaN, np.NaN
    prop = k/n
    return prop, math.sqrt(prop*(1-prop)/n)

def plot_aces_hm(mat_val, ax, title, xticks, yticks):
    im = ax.imshow(mat_val)
    cbar = ax.figure.colorbar(im, ax = ax)
    cbar.ax.set_ylabel('', rotation = -90, va = 'bottom')

    ax.set_xticks(np.arange(len(xticks)))
    ax.set_yticks(np.arange(len(yticks)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)

    ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

    ax.set_title(title)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

    for m in range(len(yticks)):
        for n in range(len(xticks)):
            text = ax.text(n, m, '{:.2f}'.format(mat_val[m, n]),
                           ha="center", va="center", color="w")


class bfs_data:
    
    ci = 0.975
    
    def __init__(self, df):
        if type(df) is str:
            self.df = pd.read_csv(df, low_memory = False)
            for ace in list(ace_list.values()):
                self.df[ace] = self.df[ace].apply(cat_code, args = (ace,))
        else:
            self.df = df
        self.keys = self.df.keys()
        
        self.corr_mat = {r:{i: None for i in income_list.keys()} 
                         for r in race_list.keys()}
        self.prop_mat = {r:{i: None for i in income_list.keys()} 
                         for r in race_list.keys()}
        
    def get_value(self, race, income, keys = []):
        ri_values = self.df[['_RACE_G1', '_INCOMG'] + list(keys)]
        
        if not race == 0:
            ri_values = ri_values[(ri_values['_RACE_G1']) == race]
            
        if not income == 0:
            ri_values = ri_values[(ri_values['_INCOMG']) == income]
        return ri_values
        
    def get_prop(self, race, income, *keys):
        
        if race not in race_list.keys() or \
            income not in income_list.keys() or\
            len(keys) == 0:
            return np.NaN, np.NaN
        
        ri_values = self.df[['_RACE_G1', '_INCOMG'] + list(keys)]
        
        if not race == 0:
            ri_values = ri_values[(ri_values['_RACE_G1']) == race]
            
        if not income == 0:
            ri_values = ri_values[(ri_values['_INCOMG']) == income]
        # ri_values = values[(values[['_RACE_G1', '_INCOMG']] == [race, income]).all(axis = 1)]
        return cal_prop(ri_values, *keys)
    
    def get_dist(self, *keys):
        
        if not keys:
            keys = list(ace_list.values())
        else:
            keys = list(keys)
        res_index = [[], []]
        res_dist = []
        for r in race_list.keys():
            for i in income_list.keys():
                prop, se = self.get_prop(r, i, *keys)
                res_dist.append([prop, se, 
                                 prop - norm.ppf(self.ci) * se, 
                                 prop + norm.ppf(self.ci) * se])
                res_index[0].append(race_list[r])
                res_index[1].append(income_list[i])
                
        return pd.DataFrame(res_dist, 
                            columns = ['Proportion', 'Standard Error',
                                      'L 95% CI', 'U 95% CI'], 
                            index = res_index)
        
    def get_corr(self, ace1, ace2 = None):
        
        res_index = [[], []]
        res_corr = []
        for r in race_list.keys():
            for i in income_list.keys():
                res_corr.append(self.get_corr_ri(r, i, ace1, ace2))
                res_index[0].append(race_list[r])
                res_index[1].append(income_list[i])
        return pd.DataFrame(res_corr, columns = ['Correlation'], index = res_index)
    
    def get_corr_ri(self, race, income, ace1, ace2 = None):
        
        if race not in race_list.keys() or\
            income not in income_list.keys():
            return np.NaN
        
        if ace2 == None:
            ace2 = ace1
        
        ri_values = self.df[['_RACE_G1', '_INCOMG'] + [ace1, ace2]]
        
        if not race == 0:
            ri_values = ri_values[(ri_values['_RACE_G1']) == race]
            
        if not income == 0:
            ri_values = ri_values[(ri_values['_INCOMG']) == income]
            
#         ace1_value = ri_values[ace1].apply(cat_code, args = (ace1,))
#         ace2_value = ri_values[ace2].apply(cat_code, args = (ace2,))
#         return ace1_value.corr(ace2_value)
        return ri_values[ace1].corr(ri_values[ace2])
    
    def __reset_mat__(self):
        self.corr_mat = {r:{i: None for i in income_list.keys()} 
                         for r in race_list.keys()}
        self.prop_mat = {r:{i: None for i in income_list.keys()} 
                         for r in race_list.keys()}
    
    def get_corr_mat(self, race, income):
        
        if not self.corr_mat[race][income] is None:
            return self.corr_mat[race][income]
        aces = list(ace_list.values())
        ri_values = self.df[['_RACE_G1', '_INCOMG'] + aces]
        
        if not race == 0:
            ri_values = ri_values[(ri_values['_RACE_G1']) == race]
            
        if not income == 0:
            ri_values = ri_values[(ri_values['_INCOMG']) == income]
        
        result = pd.DataFrame(
            [[ ri_values[ace1].corr(ri_values[ace2]) 
                   for ace1 in aces] 
                   for ace2 in aces], 
            index = aces, columns = aces)
        self.corr_mat[race][income] = result
        return result
    
    def get_prop_mat(self, race, income):
        
        if not self.prop_mat[race][income] is None:
            return self.prop_mat[race][income]
        aces = list(ace_list.values())
        ri_values = self.df[['_RACE_G1', '_INCOMG'] + aces]
        
        if not race == 0:
            ri_values = ri_values[(ri_values['_RACE_G1']) == race]
            
        if not income == 0:
            ri_values = ri_values[(ri_values['_INCOMG']) == income]
            
        result = [[ cal_prop(ri_values, ace1, ace2)
                   for ace1 in aces] 
                   for ace2 in aces]
        res_pr = [[x[0] for x in y] for y in result]
        res_se = [[x[1] for x in y] for y in result]
        result = {
            'pr': pd.DataFrame(res_pr, index = aces, columns = aces),
            'se': pd.DataFrame(res_se, index = aces, columns = aces)
                 }
        self.prop_mat[race][income] = result
        return result
        