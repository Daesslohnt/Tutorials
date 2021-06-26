import numpy as np
from numpy.core.fromnumeric import trace

def f(x):
    return 2/(1 + np.exp(-x)) - 1

def df(x):
    return 0.5*(1 + x)*(1 - x)

W1 = np.array([ 
    [-0.2, 0.3, 0.4],
    [0.1, -0.3, 0.4],
    [0.3, -0.2, 0.5] 
    ])
W2 = np.array([ 
    [0.2, -0.3, 0.4],
    [-0.6, 0.3, -0.1] 
    ])
W3 = np.array([0.2, -0.3])

def go_forward(inp):
    sum = np.dot(W1, inp)
    out1 = np.array([f(x) for x in sum])

    sum = np.dot(W2, out1)
    out2 = np.array([f(x) for x in sum])

    sum = np. dot(W3, out2)
    y = f(sum)
    return (y, out1, out2)


def train(epoch):
    lmd = 0.1
    N = 10000
    global W1, W2, W3
    count = len (epoch)
    for k in range(N):
        x = epoch[np.random.randint(0, count)]  # случайных выбор входного сигнала из обучающей выборки
        y, out1, out2 = go_forward(x[0:3])             # прямой проход по НС и вычисление выходных значений нейронов
        e = y - x[-1]                           # ошибка
        delta1 = e*df(y)                         # локальный градиент
        W3[0] = W3[0] - lmd * delta1 * out2[0]    # корректировка веса первой связи
        W3[1] = W3[1] - lmd * delta1 * out2[1]    # корректировка веса второй связи

        delta2 = W3*delta1*df(out2) 
        delta2 = np.array(delta2)              # вектор из 2-х величин локальных градиентов

        # корректировка связей первого слоя
        W2[0] = W2[0] - np.array(x[0:3]) * delta2[0] * lmd
        W2[1] = W2[1] - np.array(x[0:3]) * delta2[1] * lmd

        delta3 =  np.dot(delta2.reshape(1, 2) , W2) * df(out1)
        delta3 = np.array([x for x in delta3[0]])

        W1[0] = W1[0] - np.array(x[:3]) * delta3[0] * lmd
        W1[1] = W1[1] - np.array(x[:3]) * delta3[1] * lmd
        W1[2] = W1[2] - np.array(x[:3]) * delta3[2] * lmd

epoch = [(-1, -1, -1, -1),
         (-1, -1, 1, 1),
         (-1, 1, -1, -1),
         (-1, 1, 1, 1),
         (1, -1, -1, -1),
         (1, -1, 1, 1),
         (1, 1, -1, -1),
         (1, 1, 1, -1)]


train(epoch)

for x in epoch:
    y , out1, out2 = go_forward(x[0:3])
    print (f"Output : {y} => {x[-1]}")
    print (f'Error: {abs(y/x[-1])}')