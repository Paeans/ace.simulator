
import matplotlib.pyplot as plt
from demo import MoneyModel
import numpy as np

# wealth = []

# for i in range(100):
  # model = MoneyModel(10000)
  # for i in range(10000):
    # model.step()
    
  # wealth += [a.wealth for a in model.schedule.agents]
  
# plt.hist(wealth, bins = range(max(wealth) + 1))
# plt.show()

model = MoneyModel(50, 10, 10)
for i in range(2000):
  model.step()

gini = model.datacollector.get_model_vars_dataframe()
gini.plot()

plt.show()
  
# agent_counts = np.zeros((model.grid.width, model.grid.height))
# for cell in model.grid.coord_iter():
  # cell_content, x, y = cell
  # agent_counts[x][y] = len(cell_content)
  
# plt.imshow(agent_counts, interpolation = 'nearest')
# plt.colorbar()

# plt.show()