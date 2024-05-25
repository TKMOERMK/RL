import numpy as np
from itertools import product

aij = np.array([
    [17, 34],
    [7, 15],
    [9, 79],
    
])

bi = np.array([45, 76])

n = aij.shape[0]
m = aij.shape[1]

def calculate_deviation(aij, bi, x):
    s = np.zeros(m)
    deviation = np.zeros(m)
    d1_demand = 0.4 * bi
    d2_demand = 0.6 * bi
    d1_actual = np.zeros(m)
    d2_actual = np.zeros(m)
    for i in range(m):
        s[i] = np.sum(aij[:, i] * x)
        deviation[i] = s[i] - bi[i]
        d1_actual[i] = np.sum(aij[:, i] * (1 - x))
        d2_actual[i] = np.sum(aij[:, i] * x)
    deviation_40_60 = np.abs(d1_actual - d1_demand) + np.abs(d2_actual - d2_demand)
    return np.sum(np.abs(s - bi) + deviation_40_60)
best_x = None
min_deviation = float('inf')

for x in product([0, 1], repeat=n):
    x = np.array(x)
    deviation = calculate_deviation(aij, bi, x)
    if deviation < min_deviation:
        min_deviation = deviation
        best_x = x

print("Лучшее распределение продавцов между подразделениями (D1=0, D2=1):")
print(best_x)
print(f"Минимальное отклонение: {min_deviation}")

def print_market_division(aij, bi, x):
    print("Текущее состояние рынка:")
    for i in range(len(bi)):
        total_demand = np.sum(aij[:, i] * x)
        deviation = total_demand - bi[i]
        print(f"Продукт {i+1}: общий спрос = {total_demand}, желаемый спрос = {bi[i]}, отклонение = {deviation}")
    for j in range(len(x)):
        division = "D1" if x[j] == 0 else "D2"
        print(f"Продавец {j+1}: подразделение = {division}")

print_market_division(aij, bi, best_x)
