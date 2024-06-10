import gym
import numpy as np 
    
env = gym.make('FrozenLake8x8-v1', render_mode="human", is_slippery=False)


observation, info = env.reset(seed=42)

i = 0
for _ in range(200):
   action = 0
   if i < 7:
      action = 2
      i += 1
   elif i < 14:
      action = 1
      i += 1
   else:
      i = 0
   print(action)
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()


