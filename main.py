#!/usr/bin/python3
import build.PyRegression as pr
import numpy as np
import csv
import random
import plotly

def read_csv(filename):
    data = []
    size = 0
    max_x = 0
    with open(filename, newline='') as csvfile:
        lines = csv.reader(csvfile)
        next(lines)
        for line in lines:
            arr = np.array(line, dtype=float)
            size = len(arr) - 1
            d = pr.Data(arr[:1], arr[:-1])
            max_x = max(max_x, d.y)
            data.append(d)
    return data, size

def separate_list(data, parts = 5):
    random.shuffle(data)
    step = int(len(data) / parts)
    datalist = []
    for i in range(parts):
        datalist.append(data[i*step:(i+1)*step])

    return datalist

def run(datalist, lr):
    table_content = []
    batch_content = []
    for i in range(len(datalist)):
        data =[]
        for j in range(len(datalist)):
            if i == j: continue
            data = data + datalist[j]

        lr.train(data)
        rmse_test, r2_test = lr.calc_RMSE(datalist[i]), lr.calc_R2(datalist[i])
        rmse_train, r2_train = lr.calc_RMSE(data), lr.calc_R2(data)

        print(rmse_test, r2_test)
        batch_content.append("T" + str(i))
        table_content.append([r2_train, r2_test, rmse_train, rmse_test])

    return batch_content, table_content



data, size = read_csv('dataset.csv')
donttouch = [ size - 1 - i  for i in range(12)]
data = pr.normalize_data(data, donttouch)

lr = pr.LinearRegression(size, True)

#lr.solve_QR(data)
datalist = separate_list(data, 10)
batch, table = run(datalist, lr)

table_rows = [["R2_train", "R2_test", "RMSE_train", "RMSE_test"]] + table;

trace = plotly.graph_objs.Table( header = dict(values = (['metric'] + batch)),
                  cells=dict(values = table_rows))
                
plotly.offline.plot([trace], filename = 'basic_table')

print(lr)

