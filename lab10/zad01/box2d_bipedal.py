import gym
env = gym.make("BipedalWalker-v3", render_mode="human")

print(env.observation_space)
print(env.action_space)
observation, info = env.reset(seed=42)

for _ in range(200):
   action = env.action_space.sample()
   print(action)
   # action = [ 0.9293433,   0.25767598, -0.38051784,  0.37597746]
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()

# Box2d

# stan gry (observation space): ciągły
# zestaw akcji (action space): ciągły


# Action Space
# Box(-1.0, 1.0, (4,), float32)

# Observation Space
# Box([-3.1415927 -5. -5. -5. -3.1415927 -5. -3.1415927 -5. -0. -3.1415927 -5. -3.1415927 -5. -0. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. ], 
#     [3.1415927 5. 5. 5. 3.1415927 5. 3.1415927 5. 5. 3.1415927 5. 3.1415927 5. 5. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ], (24,), float32)