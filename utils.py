import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
import geopy.distance
import functools
import time
from itertools import permutations

def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        result = func(*args, **kwargs)
        return result
    return newfunc



