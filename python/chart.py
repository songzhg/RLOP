import pandas as pd
import matplotlib.pyplot as plt
import os

# path = 'build/examples/lunar_lander/dqn_log.txt'
path = 'build/examples/continuous_lunar_lander/ppo_log.txt'
# path = 'build/opt_results.txt'
# path = 'build/reward_results.txt'

data = pd.read_csv(path, delimiter='\t')
x = data.iloc[:, 0]
num_cols = data.shape[1] - 1

# for col in data.columns[1:]:
#     plt.plot(x, data[col], label=col)
# plt.legend()

rows = int(num_cols ** 0.5)
cols = (num_cols + rows - 1) // rows
fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
for i in range(1, data.shape[1]):
    ax = axs[(i - 1) // cols, (i - 1) % cols] 
    ax.plot(x, data.iloc[:, i], label=data.columns[i])
    ax.set_xlabel(data.columns[0]) 
    ax.set_ylabel(data.columns[i]) 
    ax.legend()
plt.tight_layout()

plt.show()