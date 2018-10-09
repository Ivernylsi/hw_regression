#!/usr/bin/python3
import build.PyRegression as pr
import numpy as np
import csv
import random
import plotly

# hot_encoding is for features which are 
#(Baset Time(0-71) N 35
#(H Local(0-23) N 39

   

def read_csv(filename):
    data = []
    size = 0
    max_x = 0
    names = []
    with open(filename, newline='') as csvfile:
        lines = csv.reader(csvfile)
        for line in lines:
            names = line[1:][:-1]
            break
        for line in lines:
            arr = np.array(line, dtype=float)[1:]
            y = arr[:-1]
            y = np.append(y, [1])
            size = len(y) 
            d = pr.Data(arr[-1:], y)
            max_x = max(max_x, d.y)
            data.append(d)
    print("Max_Y", max_x)
    return data, size, names

def separate_list(data, parts = 5):
    random.shuffle(data)
    step = int(len(data) / parts)
    datalist = []
    for i in range(parts):
        datalist.append(data[i*step:(i+1)*step])

    return datalist

def run(datalist, size):

    table_content = []
    batch_content = []
    for i in range(len(datalist)):
        data =[]
        for j in range(len(datalist)):
            if i == j: continue
            data = data + datalist[j]
        lr = pr.LinearRegression(size, False)
        lr.train(data, learn_rate = 0.05, max_iter = 5000)
        rmse_test, r2_test = lr.calc_RMSE(datalist[i]), lr.calc_R2(datalist[i])
        rmse_train, r2_train = lr.calc_RMSE(data), lr.calc_R2(data)

        print(rmse_test, r2_test)
        print(lr)
        batch_content.append("T" + str(i))
        table_content.append(np.append([r2_train, r2_test, rmse_train, rmse_test], lr.getWeight()))

    return batch_content, table_content



file1 = 'dataset.csv'
file2 = 'Dataset/Training/Features_Variant_1.csv'
data, size, names = read_csv(file1)
print(names, len(names))
donttouch = [ size - 1 - i  for i in range(12)]
donttouh = []
data = pr.normalize_data(data, donttouch)

datalist = separate_list(data, 5)
batch, table = run(datalist, size)
batch = batch + ['Mean', 'STD']
mean = [0]*(len(table[0]))
for i in range(len(table)):
    for j in range(len(table[i])):
        mean[j] += table[i][j] / len(table)

table = table + [mean]
std = [0]*len(mean)
for i in range(len(table)-1):
    for j in range(len(table[i])):
        std[j] += (table[i][j] - mean[j])**2 / len(table) 

std = np.sqrt(std)

table = table + [std] 


table_rows = [["R2_train", "R2_test", "RMSE_train", "RMSE_test"] + names] + table;

trace = plotly.graph_objs.Table( header = dict(values = (['metric'] + batch)),
                  cells=dict(values = table_rows))
                
plotly.offline.plot([trace], filename = 'basic_table')

print(lr)

