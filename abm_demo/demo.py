
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid

from mesa.datacollection import DataCollector

import matplotlib.pyplot as plt

def compute_gini(model):
  agent_wealths = [a.wealth for a in model.schedule.agents]
  x = sorted(agent_wealths)
  N = model.num_agents
  B = sum( xi * (N-i) for i,xi in enumerate(x))/(N*sum(x))
  return (1 + (1/N) - 2 * B)


class MoneyAgent(Agent):
  """ agent with fixed initial wealth """
  def __init__(self, unique_id, model):
    super().__init__(unique_id, model)
    self.wealth = 1
    
  def step(self):
    self.move()
    if self.wealth <= 0:
      return
    # other_agent = self.random.choice(self.model.schedule.agents)
    # print('agent ' + str(self.unique_id) + ' give agent ' + str(other_agent.unique_id))
    # other_agent.wealth += 1
    # self.wealth -= 1
    # print('Hi, this is agent ' + str(self.unique_id) + 
    #   ' with wealth ' + str(self.wealth))
    self.give_money()
    
  def move(self):
    possible_steps = self.model.grid.get_neighborhood(
      self.pos, moore = True, include_center = False)
    new_position = self.random.choice(possible_steps)
    self.model.grid.move_agent(self, new_position)
    
  def give_money(self):
    cellmates = self.model.grid.get_cell_list_contents([self.pos])
    if len(cellmates) > 1:
      other = self.random.choice(cellmates)
      other.wealth += 1
      self.wealth -= 1
    
class MoneyModel(Model):
  """ model with some number of agents """
  def __init__(self, N, width, height):
    self.num_agents = N
    self.schedule = RandomActivation(self)
    self.grid = MultiGrid(width, height, True)
    self.running = True
    
    for i in range(self.num_agents):
      a = MoneyAgent(i, self)
      self.schedule.add(a)
      
      x = self.random.randrange(self.grid.width)
      y = self.random.randrange(self.grid.height)
      self.grid.place_agent(a, (x,y))
      
    self.datacollector = DataCollector(
      model_reporters = {'Gini': compute_gini},
      agent_reporters = {'wealth': 'wealth'})
      
  def step(self):
    """ advance the model by one step """
    self.datacollector.collect(self)
    self.schedule.step()



