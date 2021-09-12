import numpy as np
import random
import time as tm

dice = np.arange(3425+467+469)
#dice = np.arange(10)

np.random.shuffle(dice)

print(dice)
np.save('dice9.npy', dice)
