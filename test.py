from numba import jit
import numpy as np

@jit(nopython=True)
def test_jit():
  print('a')
  array = np.arange(10)
  np.expand_dims(array,axis=0)
  #array[np.newaxis]
  sum = np.sum(array,axis=0)
  
test_jit()