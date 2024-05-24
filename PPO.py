import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback


class MarketDivisionEnv(gym.Env):
    def __init__(self, aij, bi):
        super(MarketDivisionEnv, self).__init__()
        self.aij = aij
        self.bi = bi
        self.n = aij.shape[0] 
        self.m = aij.shape[1]  
        self.action_space = gym.spaces.MultiBinary(self.n)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.m + self.n,))
    def reset(self):
        self.state = np.zeros(self.m + self.n)
        return self.state
    def step(self, action):
        x = action
        s = np.zeros(self.m)
        deviation = np.zeros(self.m)

        for i in range(self.m):
            total_demand = np.sum(self.aij[:, i] * x)
            deviation[i] = total_demand - self.bi[i]
            s[i] = np.sum(self.aij[:, i] * x)

        d1_demand = 0.4 * self.bi
        d2_demand = 0.6 * self.bi
        d1_actual = np.zeros(self.m)
        d2_actual = np.zeros(self.m)
        
        for i in range(self.m):
            d1_actual[i] = np.sum(self.aij[:, i] * (1 - x))
            d2_actual[i] = np.sum(self.aij[:, i] * x)
        
        deviation_40_60 = np.abs(d1_actual - d1_demand) + np.abs(d2_actual - d2_demand)
        reward = -np.sum(np.abs(s - self.bi) + deviation_40_60)
        self.state = np.concatenate([s, x])
        done = True
        
        return self.state, reward, done, {}

aij = np.array([
    [17, 93, 46, 2],
    [75, 44, 63, 77],
    [9, 79, 13, 73],
    [87, 12, 97, 59],
    [58, 8, 14, 43],
    [79, 95, 45, 64],
    [69, 2, 32, 75],
    [37, 15, 96, 6],
    [88, 38, 36, 5],
    [75, 15, 40, 78],
    [45, 53, 10, 71],
    [35, 88, 96, 12],
    [73, 43, 99, 30],
    [26, 26, 58, 7],
    [39, 31, 87, 69],
    [78, 77, 15, 36],
    [85, 10, 91, 73],
    [58, 77, 65, 19],
    [72, 71, 6, 15],
    [8, 22, 96, 16],
    [46, 76, 97, 84],
    [11, 41, 79, 55],
    [55, 65, 81, 32],
    [39, 93, 57, 53],
    [57, 50, 28, 43],
    [96, 69, 97, 21],
    [87, 44, 58, 73],
    [16, 61, 44, 0],
    [27, 58, 37, 59],
    [26, 63, 93, 48]
])

bi = np.array([786, 759, 888, 649])

env = MarketDivisionEnv(aij, bi)
env = DummyVecEnv([lambda: env])  

model = PPO("MlpPolicy", env, verbose=1)

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)

model.learn(total_timesteps=4500000, callback=eval_callback)
model.save("ppo_market_division")
model = PPO.load("ppo_market_division")

ideal = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0])

def print_market_division(aij, bi, x, ideal):
    print("Текущее состояние рынка:")
    for i in range(len(bi)):
        total_demand = np.sum(aij[:, i] * x)
        deviation = total_demand - bi[i]
        print(f"Продукт {i+1}: общий спрос = {total_demand}, желаемый спрос = {bi[i]}, отклонение = {deviation}")

    for j in range(len(x)):
        division = "D1" if x[j] == 0 else "D2"
        print(f"Продавец {j+1}: подразделение = {division}")

    match = np.sum(x == ideal)
    total = len(x)
    percentage_match = (match / total) * 100
    print(f"Соответствие идеальным распределением : {percentage_match:.2f}%")

obs = env.reset()
action, _states = model.predict(obs, deterministic=True)
obs, reward, done, info = env.step(action)

print_market_division(aij, bi, action[0], ideal)

def plot_results(log_folder):
    results = np.load(log_folder + '/evaluations.npz')
    timesteps = results['timesteps']
    rewards = results['results'].mean(axis=1)

    plt.figure(figsize=(12, 8))
    plt.plot(timesteps, rewards, label='Вознаграждение')
    plt.xlabel('Время обучения (шаги)')
    plt.ylabel('Вознаграждение')
    plt.title('График обучения PPO')
    plt.legend()
    plt.show()

log_folder = './logs/'
plot_results(log_folder)
