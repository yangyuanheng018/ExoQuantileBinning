import numpy as np
import random
import time as tm

group = np.load('training_dice.npy')

print(group.shape)

ps = input()

data = np.load('training_data.npz')

global_view = data['global_view']
local_view = data['local_view']
secondary_view = data['secondary_view']
scalar = data['scalar']
dispositions = data['dispositions']

print(len(global_view))
print(len(local_view))
print(len(secondary_view))
print(len(scalar))
print(len(dispositions))


dice = np.arange(len(dispositions))
#dice = np.arange(10)

np.random.shuffle(dice)

print(dice)
groups = []
for d in dice:
    groups.append(d%10)
np.save('training_dice2.npy', groups)
