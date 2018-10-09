#!/usr/bin/python3
import build.PyRegression as pr
import numpy as np
from random import randint
def func():
    a = np.array([2,3,4,5])
    data = []
    for i in range(10):
        c = np.array([randint(1,100),randint(1,100),randint(1,100),randint(1,100)])
        x = np.dot(a, c) + 100
        
        data.append(pr.Data(x, c))
    return data

data = func()
lr = pr.LinearRegression(4, True)
lr.train(data, learn_rate = 0.001, max_iter = 10000)

print(lr)

w = lr.getWeight()

#for i in data:
#    a = np.array([2,3,4,5])
#    print(np.dot(i.x,w) - i.y)
