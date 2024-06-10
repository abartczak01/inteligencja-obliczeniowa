import gym
env = gym.make('HalfCheetah-v4', render_mode="human")

print(env.observation_space)
print(env.action_space)
observation, info = env.reset(seed=42)

for _ in range(1000):
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()

# MuJoCo

