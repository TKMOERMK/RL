import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import time

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

def generate_data(n, m):
    aij = np.random.randint(1, 100, size=(n, m))
    bi = np.random.randint(50, 200, size=m)
    return aij, bi

def train_and_evaluate(n, m, target_reward):
    aij, bi = generate_data(n, m)
    env = MarketDivisionEnv(aij, bi)
    env = DummyVecEnv([lambda: env])  
    model = PPO("MlpPolicy", env, verbose=0)
    eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=500,
                                 deterministic=True, render=False)
    start_time = time.time()
    model.learn(total_timesteps=10000, callback=eval_callback)
    elapsed_time = time.time() - start_time
    results = np.load('./logs/evaluations.npz')
    rewards = results['results'].mean(axis=1)
    timesteps = results['timesteps']
    for i, reward in enumerate(rewards):
        if reward >= target_reward:
            return timesteps[i], elapsed_time
    return 10000, elapsed_time  

dimensions = [(5, 2), (30, 10), (10,30), (100,100)]
target_reward = -100  
timesteps_needed = []
times_needed = []

for n, m in dimensions:
    steps, time_needed = train_and_evaluate(n, m, target_reward)
    timesteps_needed.append(steps)
    times_needed.append(time_needed)
    print(f"Размерность (n={n}, m={m}): шагов {steps}, время {time_needed:.2f} секунд")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 2)
plt.plot([f"n={n}, m={m}" for n, m in dimensions], times_needed, marker='o')
plt.xlabel('Размерность задачи (n, m)')
plt.ylabel('Время обучения (сек)')
plt.title('Зависимость времени обучения от размерности задачи')
plt.tight_layout()
plt.show()
