import gym
env = gym.make('Acrobot-v1', render_mode="human")

print(env.observation_space)
print(env.action_space)
observation, info = env.reset(seed=42)

for i in range(400):
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()

# Classic control

# stan gry (observation space): ciągły
# zestaw akcji (action space): dyskretny

# Action Space: Discrete(3)
# Observation Space: Box([ -1. -1. -1. -1. -12.566371 -28.274334], 
#                       [ 1. 1. 1. 1. 12.566371 28.274334], (6,), float32)