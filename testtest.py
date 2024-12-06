import numpy as np


a = np.array([1 ,2, 3])
a.shape = (3, 1)
             
print(a @ np.transpose(a))