import numpy as np
import math
from scipy import linalg

ah = np.array([1,2,1])
bh = np.array([4,3,1])

l = np.cross(ah, bh)
print('l = {}'.format(l))